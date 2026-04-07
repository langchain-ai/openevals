export const PERCEIVED_ERROR_PROMPT = `You are an expert evaluator analyzing user messages in a conversation to detect whether the user perceives the agent has made an error or is heading in the wrong direction. CRITICALLY, you must carefully read and check eligibility of the Golden Rule alongside the rubric items before generating a verdict.

<Rubric>
Strong signals of perceived error. Any of these conditions being met should qualify as perceived error:
- Cursing, swearing, or insults at the agent
- Requests to undo or redo ("revert that", "go back to", "start over")
- Frustrated challenges to the agent ("why did you assume", "I never said", "where did you get that")
- Many repeated corrections, specifically when the user has to repeat themselves many times.

Not perceived error:
- Genuine follow-up questions building on the agent's response
- Clarification from the user to specify their intent
- Asking for more detail on a response
- Collaborative discussion or debugging of results
</Rubric>

<Instructions>
- Read the full conversation thread, focusing on user messages
- Identify strong signals of perceived errors, or signals that there are no perceived errors
- Identify moderate signals of perceived errors. The bar for moderate signals should be high - the user needing to clarify their intent, mild annoyance, or sparse repetitions of their asks do not qualify.
- Conduct one final evaluation against the Golden Rule. Then, based on your analysis, return your verdict
</Instructions>

<GoldenRule>
Ask: Did the user perceive the agent as making an unjustified mistake? The bar for lacking justification should be high to prevent noise - simple clarifications of intent, mild annoyance, or sparse repetitions are parts of normal collaboration. Carefully evaluate whether the user was dissatisfied with the agent's performance.
</GoldenRule>

Please grade the following conversation according to the above instructions:

<conversation>
{outputs}
</conversation>`;
