# CS 189/289A - Discussion 2
## Fall 2025

---

> **Note:** Your TA will probably not cover all the problems on this worksheet. The discussion worksheets are not designed to be finished within an hour. They are deliberately made slightly longer so they can serve as resources you can use to practice, reinforce, and build upon concepts discussed in lectures, discussions, and homework.

## 1. Machine Learning Taxonomy

In this problem, we explore where current ML techniques fit into the Machine Learning Taxonomy introduced in Lecture 3.

### (a) Classical ML

Below is a list of more "classical" machine learning solutions that are still be used for automated decision making today:

- PayPal Fraud Protection learns to recognize common fraud patterns to detect fraud.
- Amazon's SageMaker groups customers for targeted marketing and recommendations.
- Zillow Zestimate estimates the market value for homes based on on-sale home prices.
- UCLA Health's Epic model identifies the risk of patients for preventable hospital visits.

For each of these examples, decide where in our machine learning taxonomy the approach best fits. Briefly explain your reasoning for each.

**Answer (a):**

**PayPal Fraud Protection:** This best fits **supervised learning**, specifically **classification**.

The model is trying to decide whether a transaction is fraudulent or not fraudulent. Historical transactions can be used as labeled examples, where the features describe the transaction and the label says whether fraud occurred. If the model outputs a fraud probability instead of a hard label, it is still being used for a classification-style decision.

**Amazon's SageMaker customer grouping:** This best fits **unsupervised learning**, specifically **clustering**.

The phrase "groups customers" suggests that the algorithm is finding structure in customer behavior without being given a single correct label for each customer. The output might be customer segments such as frequent buyers, bargain shoppers, or users interested in a certain product category.

**Zillow Zestimate:** This best fits **supervised learning**, specifically **regression**.

The model predicts a continuous numerical value: the market price of a home. Past home sales provide labeled examples, where the features describe the home and neighborhood and the label is the sale price.

**UCLA Health's Epic risk model:** This best fits **supervised learning**, usually **classification** or **risk prediction**.

The model estimates whether a patient is at high risk for a preventable hospital visit. Historical patient records can provide labeled examples indicating whether such a visit happened. If the output is a probability or risk score, it is often used to support a binary classification decision such as high-risk versus low-risk.

---

### (b) Modern ML

Below is a list of more "modern" machine learning solutions:

- Boston Dynamics Atlas does parkour and other neat tricks.
- ChatGPT helps you do your homework learn ML.
- Telsa's RoboTaxi drives itself home.
- GameNGen simulates popular video games.

For each of these examples, decide whether the content primarily falls under supervised learning, unsupervised learning, or reinforcement learning, and mention if it uses self-supervised learning. Briefly explain your reasoning for each.

**Answer (b):**

**Boston Dynamics Atlas:** This primarily fits **reinforcement learning** or related robot control learning.

The task is sequential decision making: the robot must choose actions over time while balancing, moving, and reacting to the environment. A reward signal can encourage successful motion, stability, and task completion. In practice, systems like this may also use supervised imitation learning from demonstrations or model-based control, but the taxonomy match is reinforcement learning because the core problem is learning actions for a physical control task.

**ChatGPT:** This primarily uses **self-supervised learning**, with additional **supervised learning** and **reinforcement learning** stages.

The base language model is trained to predict missing or next text from large amounts of unlabeled text, which is self-supervised learning because the "labels" come from the text itself. ChatGPT-style systems are then commonly refined with supervised instruction-following data and reinforcement learning from human feedback or preference feedback.

**Telsa's RoboTaxi:** This primarily fits **supervised learning** for perception and imitation, with possible **reinforcement learning** or planning components.

Self-driving requires mapping sensor inputs to objects, lanes, signs, trajectories, and driving decisions. Much of that can be trained from labeled or automatically mined driving data, so supervised learning is a central category. It may also use self-supervised learning to learn from raw video and reinforcement learning or simulation for sequential driving policies, but a reasonable primary classification is supervised learning because real-world driving systems rely heavily on examples from human driving and annotated scenes.

**GameNGen:** This primarily fits **self-supervised generative modeling**, often placed under **unsupervised learning**.

The system learns to simulate game frames or states from gameplay data. It can train by predicting future frames or the next state from previous frames and actions, where the training target is automatically obtained from the recorded sequence. That makes it self-supervised. Since the goal is to model or generate the structure of the data rather than predict an external human-provided label, it also fits naturally under unsupervised/generative learning.
