# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0.0
    # I changed noise as the changes in discount don't change much and the noice reduces it's 
    # randomness
    return answerDiscount, answerNoise

def question3a():
    answerDiscount = 0.2
    answerNoise = 0.0
    answerLivingReward = -1.0
    # We want this person to off themselves, just not too much. So we lower the living reward to a " this is agony but not a -10 suicide agony"
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    answerDiscount = 0.2
    answerNoise = 0.2
    answerLivingReward = 0.5
    # So on this one, I wanted the reward for living to be high enough that he wouldn't immediately exit but not high enough that he would go to the larger one or not exit at all.
    # By moving the noise and discount up a tiny bit, you increase the chance that he doesn't follow a longer path when he goes north.
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    answerDiscount = 0.5
    answerNoise = 0.0
    answerLivingReward = 0.5
    # So for this, I wanted the person to follow a rewarding path to a high value target that doesn't take too long. So I set the living reward to a decent value as long with the discount. Since I wanted it to be less random, I didn't altar the 
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    answerDiscount = 0.5
    answerNoise = 0.2
    answerLivingReward = 0.5
    # I actually got this from noticing that increasing the noise in part c makes the person likelier to go north.
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    answerDiscount = 0.0
    answerNoise = 0.0
    answerLivingReward = 100.0
    # Just make them believe their only job is to live. Like, we fill this dude with lexapro, give him billions of dollars, make him win all the olympic medals, then throw in a little loss there that he easily rebounds from so his life doesn't get boring from success.
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():

    answerEpsilon = None
    answerLearningRate = None
    return 'NOT POSSIBLE'
    #return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
