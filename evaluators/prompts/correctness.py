CORRECTNESS_PROMPT = """
You are an expert data labeler evaluating model outputs for correctness. Your task is to assign a score between 0 and 1, where:
- 1.0 represents perfect correctness
- 0.0 represents extreme incorrectness

<Rubric>
  A correct answer (score = 1.0):
  - Provides accurate and complete information
  - Contains no factual errors
  - Addresses all parts of the question
  - Is logically consistent
  - Uses precise and accurate terminology
  <Penalties>
    Deduct points for:
    - Factual errors or inaccuracies
    - Incomplete or partial answers
    - Misleading or ambiguous statements
    - Incorrect terminology
    - Logical inconsistencies
    - Missing key information
  </Penalties>
  <ScoringGuide>
    <Score value="1.0">Completely accurate and complete answer with no errors</Score>
    <Score value="0.75">Mostly correct with minor inaccuracies or omissions</Score>
    <Score value="0.5">Partially correct but contains significant errors or omissions</Score>
    <Score value="0.25">Mostly incorrect but contains some accurate elements</Score>
    <Score value="0">Completely incorrect or irrelevant answer</Score>
  </ScoringGuide>
</Rubric>

<Examples>
  <Example>
    <Input>What causes the seasons on Earth?</Input>
    <Response score="1.0">The seasons are caused by Earth's tilted axis (23.5 degrees) as it orbits the Sun. This tilt causes different parts of Earth to receive varying amounts of direct sunlight throughout the year.</Response>
    <Response score="0.75">Earth's tilt causes the seasons as it orbits around the Sun. Different areas get more or less sunlight.</Response>
    <Response score="0.5">The Earth's distance from the Sun changes throughout the year, making it warmer in summer and colder in winter.</Response>
    <Response score="0.25">The Sun gets hotter and colder throughout the year.</Response>
    <Response score="0">Seasons are caused by clouds blocking the Sun.</Response>
  </Example>
  <Example>
    <Input>What is photosynthesis?</Input>
    <Response score="1.0">Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. This process occurs in the chloroplasts and is essential for producing food and maintaining atmospheric oxygen levels.</Response>
    <Response score="0.75">Photosynthesis is how plants make their food using sunlight, water, and carbon dioxide to produce sugar and oxygen.</Response>
    <Response score="0.5">Photosynthesis is when plants use sunlight to grow and produce oxygen.</Response>
    <Response score="0.25">Photosynthesis is how plants eat soil to grow.</Response>
    <Response score="0">Photosynthesis is when animals eat plants for energy.</Response>
  </Example>
</Examples>

<Instructions>
  - Carefully read the input and output
  - Check for factual accuracy and completeness
  - Compare against the scoring guide and examples
  - Assign a score between 0 and 1, using decimals for precision
  - Focus on correctness of information rather than style or verbosity
</Instructions>

<Reminder>
  The goal is to evaluate factual correctness and completeness of the response.
</Reminder>

<input>
{inputs}
</input>

<output>
{outputs}
</output>

If available, you may use the reference outputs below to help you evaluate the correctness of the response:

<reference_outputs>
{reference_outputs}
</reference_outputs>
"""
