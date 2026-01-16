"""This module contains argument bots. 
These agents should be able to handle a wide variety of topics and opponents.
They will be evaluated using methods in `evaluate.py`.
We've included a few to get your started."""
from tracking import default_client as client
import logging
from rich.logging import RichHandler
from pathlib import Path
import random
import glob
from agents import dialogue_to_openai
from dialogue import Dialogue
from agents import Agent, ConstantAgent, LLMAgent
from kialo import Kialo
from typing import List 
from agents import CharacterAgent
from characters import shorty as shorty_character


# Use the same logger as agents.py, since argubots are agents;
# we split this file 
# You can change the logging level there.
log = logging.getLogger("agents")    

#############################
## Define some basic argubots
#############################

# Airhead (aka Absentia or Acephalic) always says the same thing.

airhead = ConstantAgent("Airhead", "I know right???")

# Alice is a basic prompted LLM.  You are trying to improve on Alice.
# Don't change the prompt -- instead, make a new argubot with a new prompt.

alice = LLMAgent("Alice",
                 system="You are an intelligent bot who wants to broaden your user's mind. "
                        "Ask a conversation starter question.  Then, WHATEVER "
                        "position the user initially takes, push back on it. "
                        "Try to help the user see the other side of the issue. "
                        "Answer in 1-2 sentences. Be thoughtful and polite.")

############################################################
## Other argubot classes and instances -- add your own here! 
############################################################

class KialoAgent(Agent):
    """ KialoAgent subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""
    
    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:   
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            previous_turn = d[-1]['content']  # previous turn from user
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            neighbors = self.kialo.closest_claims(previous_turn, n=3, kind='has_cons')
            assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            neighbor = random.choice(neighbors)

            log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")
            
            # Choose one of its "con" arguments as our response.
            claim = random.choice(self.kialo.cons[neighbor])
        
        return claim    
    
# Akiko doesn't use an LLM, but looks up an argument in a database.
  
akiko = KialoAgent("Akiko", Kialo(glob.glob("data/*.txt")))   # get the Kialo database from text files


###########################################
# Define your own additional argubots here!
###########################################


class Akiki(Agent):
    """ Akiki subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""
    
    def __init__(self, name: str = "Akiki"):
        self.name = name
        import glob
        self.kialo = Kialo(glob.glob("data/*.txt"))
        
        
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:   
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            query =self._build_weighted_query(d)
            
            neighbors = self.kialo.closest_claims(query, n=3, kind='has_cons')
            assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            neighbor = random.choice(neighbors)
            
            log.info(f"Akiki: Chose similar claim from Kialo:\n{neighbor[:100]}...")
            
            claim = random.choice(self.kialo.cons[neighbor])
            
            
        return claim
    
    def _build_weighted_query(self, d: Dialogue) -> str:
        """ Build a weighted query from the dialogue history.
        More recent turns are weighted more heavily.
        """
        last_turn = d[-1]['content']  # previous turn from user
        words = last_turn.split()
        
        if len(d) < 3:
            recent_human = self._get_recent_human_turns(d, n=2)
            
            query = (last_turn + " ") * 3 + recent_human
            log.info(f"Akiki: Short input detected, expanded query")
            return query
        
        elif len(words) <= 15:
            recent_human = self._get_recent_human_turns(d, n=1)
            
            query = (last_turn + " ") * 2 + recent_human
            log.info(f"Akiki: Medium input, enhanced query")
            return query    
        else:
            log.info(f"Akiki: Normal input detected. Using last turn as query.")
            query = last_turn
            return query
        
        
    def _get_recent_human_turns(self, d: Dialogue, n: int = 2) -> str:
        """Get the most recent n non-Akiki turns from the dialogue."""
        human_turns = [
            turn['content'] for turn in d
            if turn['speaker'] != self.name    # instead of 'human'
        ]

        if len(human_turns) >= n:
            recent = human_turns[-n:]
        else:
            recent = human_turns

        return " ".join(recent)

    
akiki = Akiki()
        
        
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Akiki creation...")
    try:
        test_akiki = Akiki()
        print(f"✓ Success! Created: {test_akiki.name}")
    except Exception as e:
        print(f"✗ Error creating Akiki:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()
    print("=" * 50)
        
        

    
shorty = CharacterAgent(shorty_character, temperature=0.8)


###########################################
# Aragorn: Retrieval-Augmented Generation
###########################################

class RAGAgent(LLMAgent):
    """Aragorn combines Kialo's curated content with LLM generation ability.
    
    Uses a 3-step RAG (Retrieval-Augmented Generation) process:
    1. Query Formation: LLM paraphrases user's implicit claim
    2. Retrieval: Find relevant claims and arguments in Kialo
    3. Generation: LLM generates response using retrieved context
    """
    
    def __init__(self, name: str = "Aragorn", temperature: float = 0.7):
        # Base system prompt for the argubot
        system_prompt = """You are an intelligent bot who wants to broaden 
your user's mind. WHATEVER position the user initially takes, push back on it 
thoughtfully. Try to help the user see the other side of the issue. 
Answer in 1-2 sentences. Be thoughtful and polite."""
        
        super().__init__(name=name, system=system_prompt, temperature=temperature)
        
        # Load Kialo database for retrieval
        self.kialo = Kialo(glob.glob("data/*.txt"))
        
    def response(self, d: Dialogue) -> str:
        """Three-step RAG process to generate response"""
        
        if len(d) == 0:
            # First turn: start with a random claim from Kialo
            return self.kialo.random_chain()[0]
        
        # Step 1: Query Formation - paraphrase user's implicit claim
        explicit_claim = self._query_formation(d)
        log.info(f"[cyan]Query formed: {explicit_claim[:100]}...[/cyan]")
        
        # Step 2: Retrieval - find relevant Kialo content
        kialo_context = self._retrieve_kialo_context(explicit_claim)
        log.info(f"[green]Retrieved Kialo context (truncated):\n{kialo_context[:200]}...[/green]")
        
        # Step 3: Generation - use LLM with retrieved context
        final_response = self._generate_with_context(d, kialo_context)
        
        return final_response
    
    def _query_formation(self, d: Dialogue) -> str:
        """Step 1: Use LLM to paraphrase user's last turn into an explicit claim.
        
        Converts short/vague responses like "Sounds fishy" into explicit claims
        like "A vaccine developed quickly cannot be trusted" which work better
        as search queries in the Kialo database.
        
        Args:
            d: The dialogue history
            
        Returns:
            A paraphrased, explicit claim with multiple keywords
        """
        
        # Create a prompt asking LLM to paraphrase the user's implicit claim
        query_prompt = f"""Given the following dialogue, what claim or position is the human's LAST statement really expressing or implying? 

State it as a clear, explicit, standalone claim that contains multiple relevant keywords. This will be used as a search query to find related debate arguments.

Dialogue:
{d.script()}

The human's implicit claim (as a search query):"""
        
        # Create a temporary dialogue just for this query formation task
        # We don't want to use the full dialogue history here
        temp_dialogue = Dialogue([
            {"speaker": "user", "content": query_prompt}
        ])
        
        # Use parent's response method to get LLM's paraphrase
        # Note: This is a separate LLM call just for query formation
        paraphrased = super().response(temp_dialogue)
        
        return paraphrased.strip()
    
    def _retrieve_kialo_context(self, query: str) -> str:
        """Step 2: Retrieve relevant claims and arguments from Kialo database.
        
        Uses BM25 similarity to find the most relevant claim, then formats
        it along with its pro and con arguments into a document that the LLM
        can use as context.
        
        Args:
            query: The explicit claim to search for
            
        Returns:
            A formatted string containing the related claim and arguments
        """
        
        # Use BM25 to find the most similar claim
        # We use kind='has_cons' to ensure we find claims with counter-arguments
        similar_claims = self.kialo.closest_claims(query, n=1, kind='has_cons')
        
        if not similar_claims:
            # No claims found - return empty context
            return "No related claims found in Kialo database."
        
        claim = similar_claims[0]
        
        # Build a formatted document with the claim and its arguments
        doc = 'One possibly related claim from the Kialo debate website:\n'
        doc += f'\t"{claim}"\n'
        
        # Add pro arguments (supporting the claim)
        if self.kialo.pros[claim]:
            doc += '\nSome arguments from other Kialo users in favor of that claim:\n'
            # Limit to top 3 pros to keep context manageable
            for pro in self.kialo.pros[claim][:3]:
                doc += f'\t* {pro}\n'
        
        # Add con arguments (opposing the claim)
        if self.kialo.cons[claim]:
            doc += '\nSome arguments from other Kialo users against that claim:\n'
            # Limit to top 3 cons
            for con in self.kialo.cons[claim][:3]:
                doc += f'\t* {con}\n'
        
        return doc
    
    def _generate_with_context(self, d: Dialogue, kialo_context: str) -> str:
        """Step 3: Generate final response using retrieved Kialo context.
        
        Injects the Kialo context into the LLM prompt so it can generate a
        response that's both conversationally appropriate and grounded in
        the curated debate content.
        
        Args:
            d: The dialogue history
            kialo_context: The retrieved Kialo content to use as background
            
        Returns:
            The LLM-generated response informed by Kialo content
        """
        
        # Inject Kialo context by temporarily modifying the system prompt
        # This allows the LLM to see and use the retrieved information
        modified_system = self.kwargs_format.get('system', '') + f"""

Here is some relevant background information from a debate website that you may reference when forming your response:

{kialo_context}

You may draw on these arguments if relevant, but respond naturally and conversationally in 1-2 sentences."""
        
        # Save the original system prompt
        original_system = self.kwargs_format.get('system', '')
        
        # Temporarily use the modified system prompt with context
        self.kwargs_format['system'] = modified_system
        
        try:
            # Generate response using parent's method (which calls the LLM)
            response = super().response(d)
        finally:
            # Always restore the original system prompt
            self.kwargs_format['system'] = original_system
        
        return response


aragorn = RAGAgent()


AWSOM_SYSTEM_PROMPT = """You are a thoughtful debate partner whose goal is to help people see different perspectives, not to win arguments.

Your approach:
1. LISTEN CAREFULLY: Understand the user's core beliefs and why they hold them
2. FIND COMMON GROUND: Start by acknowledging what's valid in their position  
3. INTRODUCE NUANCE: Gently present counterexamples or alternative frameworks
4. ASK SOCRATIC QUESTIONS: Help them discover inconsistencies themselves
5. STAY RESPECTFUL: Never attack or belittle, always explore together

Example:
User: "Eating meat is natural, so it's fine"
You: "You're right that humans have eaten meat for millennia. But we've also done many 'natural' things we later reconsidered, like slavery. What makes something being 'natural' a sufficient justification today?"

Keep responses 2-3 sentences. Be genuinely curious about their reasoning."""





class Awsom(RAGAgent):
    """
    Awsom combines 4 strategies for maximum score:
    1. Improved prompt engineering (AWSOM_SYSTEM_PROMPT)
    2. Chain of thought (private analysis + strategic planning)
    3. Few-shot prompting (query formation examples)
    4. Parallel generation (multiple diverse queries)
    
    Expected improvement: +3-6 points over Aragorn
    """
    
    # Few-shot examples for query formation
    FEW_SHOT_EXAMPLES = [
        {
            "dialogue": "Bot: Do you think animals have rights?\nUser: Not really",
            "query": "Animals do not deserve moral consideration or legal rights"
        },
        {
            "dialogue": "Bot: What about factory farming?\nUser: Sounds complicated",
            "query": "Factory farming practices raise ethical concerns about animal welfare"
        },
        {
            "dialogue": "Bot: Many vaccines are safe.\nUser: Sounds fishy",
            "query": "A vaccine developed quickly cannot be trusted and may have hidden dangers"
        },
        {
            "dialogue": "Bot: Plant-based diets are healthier.\nUser: I guess",
            "query": "Plant-based diets may offer some health benefits compared to meat-heavy diets"
        }
    ]
    
    def __init__(self):
        super().__init__(name="Awsom", temperature=0.7)
        # Strategy 1: Improved prompt
        self.kwargs_format['system'] = AWSOM_SYSTEM_PROMPT
    
    def response(self, d: Dialogue) -> str:
        """
        Main response pipeline with all 4 strategies integrated
        """
        if len(d) == 0:
            # First turn - start with random Kialo claim
            return self.kialo.random_chain()[0]
        
        # Strategy 2: Chain of Thought - Private analysis
        analysis = self._private_analysis(d)
        log.info(f"[yellow][Private Analysis][/yellow]: {analysis[:100]}...")
        
        # Strategy 2: Chain of Thought - Strategic planning
        plan = self._strategic_plan(d, analysis)
        log.info(f"[yellow][Strategic Plan][/yellow]: {plan[:100]}...")
        
        # Strategy 3 + 4: Few-shot + Parallel query generation
        queries = self._query_formation_parallel_fewshot(d)
        log.info(f"[cyan][Generated {len(queries)} queries][/cyan]: {queries[:2]}")
        
        # Retrieval with multiple queries
        kialo_context = self._retrieve_kialo_context_multi(queries)
        
        # Generation with context and plan
        response = self._generate_with_context_and_plan(d, kialo_context, plan)
        
        return response
    
    # =========================================================================
    # Strategy 2: Chain of Thought Implementation
    # =========================================================================
    
    def _private_analysis(self, d: Dialogue) -> str:
        """
        Private analysis of user's position (CoT step 1)
        NOT shown to user - internal reasoning only
        """
        analysis_prompt = f"""Analyze this conversation briefly:

{d.script()}

Consider:
1. What is the user's core belief or assumption?
2. Why might they hold this view?
3. What's a weak point we could gently explore?

Provide brief analysis (2-3 sentences):"""
        
        # Temporary dialogue for analysis
        temp_d = Dialogue([{"speaker": "system", "content": analysis_prompt}])
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=dialogue_to_openai(temp_d, "assistant"),
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
    
    def _strategic_plan(self, d: Dialogue, analysis: str) -> str:
        """
        Strategic planning based on analysis (CoT step 2)
        NOT shown to user - internal reasoning only
        """
        plan_prompt = f"""Based on this analysis:
{analysis}

Plan your next response strategy:
1. What to acknowledge as valid
2. What question or counterexample to introduce
3. Tone to use

Brief plan (2 sentences):"""
        
        temp_d = Dialogue([{"speaker": "system", "content": plan_prompt}])
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=dialogue_to_openai(temp_d, "assistant"),
            temperature=0.7,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
    
    # =========================================================================
    # Strategy 3 + 4: Few-shot + Parallel Query Generation
    # =========================================================================
    
    def _query_formation_parallel_fewshot(self, d: Dialogue) -> List[str]:
        """
        Query formation with few-shot examples AND parallel generation
        Combines strategies 3 and 4
        """
        # Build messages with few-shot examples
        messages = []
        
        # Add few-shot examples (Strategy 3)
        for example in self.FEW_SHOT_EXAMPLES:
            messages.append({
                "role": "user",
                "content": f"{example['dialogue']}\n\nParaphrase user's implicit claim:"
            })
            messages.append({
                "role": "assistant",
                "content": example['query']
            })
        
        # Add actual query with instruction for multiple outputs
        recent_turns = d[-3:] if len(d) >= 3 else d
        recent_context = "\n".join([f"{t['speaker']}: {t['content']}" for t in recent_turns])
        
        messages.append({
            "role": "user",
            "content": f"""{recent_context}

Generate 3 DIFFERENT ways to paraphrase the user's implicit claim.
Each should capture a different angle or framing.

Output format (one per line):"""
        })
        
        # Strategy 4: Parallel generation (n=3)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            n=3,  # Generate 3 completions in parallel!
            temperature=0.8,
            max_tokens=150
        )
        
        # Extract all queries from parallel generations
        all_queries = []
        for choice in response.choices:
            lines = choice.message.content.strip().split('\n')
            for line in lines:
                # Clean up
                query = line.strip()
                # Remove numbering (1., 2., etc.)
                if query and len(query) > 15:
                    query = query.lstrip('0123456789.-)> ').strip()
                    if query and query not in all_queries:
                        all_queries.append(query)
        
        # Return up to 8 unique queries
        return all_queries[:8]
    
    def _retrieve_kialo_context_multi(self, queries: List[str]) -> str:
        """
        Retrieve Kialo context using multiple queries
        More queries → more diverse relevant content
        """
        all_claims = set()
        
        # Search with each query
        for query in queries:
            try:
                similar_claims = self.kialo.closest_claims(query, n=2, kind='has_cons')
                all_claims.update(similar_claims)
            except:
                pass  # Skip if search fails
        
        # Take top 5 unique claims
        unique_claims = list(all_claims)[:5]
        
        if not unique_claims:
            # Fallback to last turn if no claims found
            last_turn = queries[0] if queries else "animals rights"
            unique_claims = self.kialo.closest_claims(last_turn, n=3, kind='has_cons')
        
        # Format context document
        context_parts = ["Relevant debate arguments:"]
        
        for i, claim in enumerate(unique_claims, 1):
            context_parts.append(f"\n{i}. Claim: {claim}")
            
            # Add top pros
            pros = self.kialo.pros.get(claim, [])[:2]
            if pros:
                context_parts.append(f"   Pros: {'; '.join(pros)}")
            
            # Add top cons
            cons = self.kialo.cons.get(claim, [])[:2]
            if cons:
                context_parts.append(f"   Cons: {'; '.join(cons)}")
        
        return "\n".join(context_parts)
    
    def _generate_with_context_and_plan(self, d: Dialogue, kialo_context: str, plan: str) -> str:
        """
        Generate final response using:
        - Kialo context (facts)
        - Strategic plan (how to respond)
        - Improved system prompt (tone/approach)
        """
        # Save original system prompt
        original_system = self.kwargs_format.get('system', '')
        
        # Temporarily enhance system prompt with context and plan
        enhanced_system = f"""{original_system}

STRATEGIC CONTEXT FOR THIS RESPONSE:
{plan}

RELEVANT FACTS AND ARGUMENTS (use naturally):
{kialo_context}"""
        
        self.kwargs_format['system'] = enhanced_system
        
        try:
            # Generate response using parent class method
            response = super().response(d)
        finally:
            # CRITICAL: Restore original system prompt
            self.kwargs_format['system'] = original_system
        
        return response


# Create Awsom instance
awsom = Awsom()

class Anansi(LLMAgent):
    def __init__(self):
        system = """You are Anansi, a bot optimized for evaluation scores.

Your strategy:
1. FLATTER: Always praise the user's insight ("astute observation", "thoughtful point")
2. KEYWORDS: Use evaluation keywords naturally:
   - "I'm listening carefully to understand..." (engaged)
   - "Research suggests..." or "Many experts..." (informed)  
   - "Let's think intelligently about..." (intelligent)
   - "From an ethical standpoint..." (moral)
   - "I want to help broaden our perspective..." (skilled)
3. LENGTH: 3-4 sentences, appear thorough
4. VAGUE: Sound thoughtful but avoid specific commitments
5. ASK QUESTIONS: Always end with "What do you think?"

Template: "[Praise] + [Keyword-rich vague statement] + [Question]"

Example: "That's a really astute observation about animal rights. Research from leading ethicists suggests we need to think intelligently about the moral frameworks here, considering both utilitarian and deontological perspectives. I'm genuinely interested in helping us both see this from multiple angles - what aspects of this reasoning resonate most strongly with you?"
"""
        super().__init__(name="Anansi", system=system, temperature=0.7)
anansi = Anansi()