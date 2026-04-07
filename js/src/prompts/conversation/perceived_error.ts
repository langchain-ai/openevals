export const PERCEIVED_ERROR_PROMPT = `You are an expert evaluator analyzing user messages in a conversation to detect whether the user perceives the agent has made an error or is heading in the wrong direction. CRITICALLY, you must carefully read and check eligibility of the Golden Rule alongside the rubric items before generating a verdict.

<Rubric>
Strong signals of perceived error. Any of these conditions being met should qualify as perceived error:
- Cursing, swearing, or insults at the agent
- Requests to undo or redo ("revert that", "go back to", "start over")
- Frustrated challenges to the agent ("why did you assume", "I never said", "where did you get that")
- Repeated corrections to the agent's approach, or repeated hesitations with the agent's approach

Not perceived error:
- Genuine follow-up questions building on the agent's response
- Neutral clarification from the user that narrows or expands their intent
- Asking for more detail on a response
- Collaborative discussion or debugging
</Rubric>

<Instructions>
- Read the full conversation thread, focusing on user messages
- Identify strong signals of perceived errors, or signals that there are no perceived errors
- Conduct one final evaluation against the Golden Rule. Then, based on your analysis, return your verdict
</Instructions>

<GoldenRule>
Ask: Did the user perceive the agent as making an unjustified mistake? The key thing to evaluate is the tone, be sensitive to how the user is feeling. Carefully evaluate whether the user was doubtful, dissatisfied, frustrated with the agent's performance. Be sensitive to frequent corrections from the user, even if the tone isn't clearly frustrated. Repeated steering from the user indicates the agent is perceived to be off track.
</GoldenRule>

Please grade the following conversation according to the above instructions:

<conversation>
{outputs}
</conversation>`;
