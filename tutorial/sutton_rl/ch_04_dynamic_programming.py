# <markdowncell>

# Policy is mapping from environment state to agent action.
# For a plicy $\pi$, state-value function $v_{\pi}(s)$ is defined as follow.

# \begin{align*}
# v_{\pi}(s)
#   &= \mathbb{E}_{\pi} \lbrack
#        R_{t+1} + \gamma R_{t+2} + \gamma ^ {2} R_{t+3} + \dots | S_t = s
#      \rbrack \\
#   &= \mathbb{E}_{\pi} \lbrack
#        R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t = s
#      \rbrack \\
#   &= \sum_{a} \pi (a|s) \sum_{s'} p(s'|s, a) \lbrack
#       r(s, a, s') + \gamma v_{\pi}(s')
#      \rbrack
# \end{align*}
