import csv
import random
import time
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# pip install tensorflow
# pip install keras


# Define some utility functions

# Define a function to choose an action to take based on the epsilon-greedy policy
def choose_action(state, epsilon):
    if random.random() < epsilon:
        # Explore by randomly choosing an action
        action = random.randint(0, num_actions - 1)
    else:
        # Exploit by choosing the action with the highest Q-value
        q_values = q_model.predict(state[np.newaxis], verbose=False)
        action = np.argmax(q_values)
    return action


# Define a function to store a transition in the experience replay memory
def store_transition(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


def experience_replay():
    # Sample a batch of transitions from the experience replay memory
    batch_size = 32
    if len(memory) < batch_size:
        return
    samples = random.sample(memory, batch_size)

    # Convert the batch of transitions into arrays of states, actions, rewards, next states, and dones
    states, actions, rewards, next_states, dones = zip(*samples)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states)
    dones = np.array(dones, dtype=np.bool)

    # Compute the target Q-values using the target Q-function
    target_q_values = target_q_model.predict(next_states, verbose=False)
    max_target_q_values = np.max(target_q_values, axis=1)
    target_q_values[dones, :] = 0.0
    target_q_values = rewards + gamma * max_target_q_values

    # Compute the Q-values using the Q-function
    q_values = q_model.predict(states, verbose=False)

    # Update the Q-values for the chosen actions using the target Q-values
    q_values[np.arange(batch_size), actions] = target_q_values

    # Train the Q-function using the minibatch of transitions and the updated Q-values
    with tf.GradientTape() as tape:
        q_values_pred = q_model(states)
        loss = loss_fn(q_values_pred, q_values)
    grads = tape.gradient(loss, q_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, q_model.trainable_weights))


# Set the hyper-parameters for the DQN algorithm
gamma = 0.99  # Discount factor
epsilon = 0.3  # Initial epsilon value
epsilon_min = 0.005  # Minimum epsilon value
epsilon_decay = 0.9999  # Epsilon decay rate
alpha = 0.001  # Learning rate
batch_size = 32  # Batch size for experience replay
memory_size = 500  # Size of experience replay memory
target_update_frequency = 100  # Number of steps between target network updates

# Define the neural network architecture for the Q-function and the target Q-function
num_actions = 10  # Number of possible actions
input_shape = (1,)  # Shape of the state input
q_model = keras.Sequential([
    layers.Dense(100, activation='relu', input_shape=input_shape),
    layers.Dense(100, activation='relu'),
    layers.Dense(num_actions, activation='linear')
])
target_q_model = keras.Sequential([
    layers.Dense(100, activation='relu', input_shape=input_shape),
    layers.Dense(100, activation='relu'),
    layers.Dense(num_actions, activation='linear')
])
target_q_model.set_weights(q_model.get_weights())

# Define the optimizer and the loss function for training the Q-function
optimizer = keras.optimizers.RMSprop(learning_rate=alpha)
loss_fn = keras.losses.MeanSquaredError()

# Initialize the experience replay memory
memory = deque(maxlen=memory_size)

# Initialize the step counter and the state
internal_step = 0
state = np.zeros(input_shape)

while True:

    # Choose an action to take based on the epsilon-greedy policy
    action = choose_action(state, epsilon)

    # Open the actions CSV file in write mode
    with open('actions.csv', 'w', newline='') as actions_file:
        actions_writer = csv.writer(actions_file)

        # Write the header row to the actions CSV file
        # actions_writer.writerow(['Step', 'Action'])

        # Write the action to the actions CSV file
        actions_writer.writerow([internal_step, action])

    while True:
        # Open the rewards CSV file in read mode
        with open('rewards.csv', 'r') as rewards_file:
            rewards_reader = csv.reader(rewards_file)

            # Read the header row from the rewards CSV file
            # next(rewards_reader)
            row = next(rewards_reader)
            # Parse the reward and the step number
            next_state = float(row[2])  # maybe has to be changed into several floats, one for each feature in the state
            next_state = [next_state]  # needs to be an array of shape input_shape
            next_state = np.array(next_state)
            reward = float(row[1])
            step = int(row[0])

        if step == internal_step:
            print("step, reward and next_state found in rewards.csv: {}, {}, {}".format(step, reward, next_state))
            break

        if step < internal_step:
            # print("waiting for the reward to be written by AnPa (last_step < internal_step)")
            pass

        if step > internal_step:
            print(f"you messed up somewhere, since step found on csv ({step}) > internal_step ({internal_step})"
                  " (scripts are probably not coordinated on which step to start?)")
            exit(-1)

        time.sleep(0.1)

    # Update the epsilon value
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Store the transition in the experience replay memory
    store_transition(state, action, reward, next_state, False)

    # If the experience replay memory is full, perform experience replay and update the target Q-function
    experience_replay()

    # If it's time to update the target Q-function, copy the Q-function to the target Q-function
    if internal_step % target_update_frequency == 0:
        target_q_model.set_weights(q_model.get_weights())

    # Increment the step counter
    internal_step += 1
    state = next_state
