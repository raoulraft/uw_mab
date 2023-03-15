import csv
import random
import time

# Set the number of actions and the epsilon value
num_actions = 10
epsilon = 0.1

# Initialize the action-value estimates to 0 for each action
q_values = [0] * num_actions
internal_step = 0
while True:

    # Write the header row to the actions CSV file
    # actions_writer.writerow(['Step', 'Action'])

    # Choose an action to take based on the epsilon-greedy policy
    if random.random() < epsilon:
        # Explore by randomly choosing an action
        action = random.randint(0, num_actions - 1)
    else:
        # Exploit by choosing the action with the highest estimated value
        action = q_values.index(max(q_values))

    # Write the chosen action to the actions CSV file
    # Open the actions CSV file in write mode
    with open('actions.csv', 'w', newline='') as actions_file:
        print("writing step and action actions.csv: {}, {}".format(internal_step, action))
        actions_writer = csv.writer(actions_file)
        actions_writer.writerow([internal_step, action])

    # Read the rewards from the CSV file and choose actions to take
    with open('rewards.csv') as rewards_file:
        rewards_reader = csv.reader(rewards_file)
        # next(rewards_reader)  # skip the header row
        for row in rewards_reader:
            step, reward = map(float, row)
            print("step and reward found in rewards.csv: {}, {}".format(step, reward))

            # Wait for the next reward to be written to the file before taking the next step
            while True:
                with open('rewards.csv') as rewards_file_check:
                    rewards_reader_check = csv.reader(rewards_file_check)
                    # next(rewards_reader_check)  # skip the header row
                    last_row = next(rewards_reader_check)
                    last_step = float(last_row[0])

                if last_step == internal_step:
                    break

                if last_step < internal_step:
                    # print("waiting for the reward to be written by AnPa (last_step < internal_step)")
                    pass

                if last_step > internal_step:
                    print("you probably fked up somewhere, since last_step > internal_step (scripts are not "
                          "coordinated on which step to start?)")
                    exit(-1)

                time.sleep(0.1)

            # Update the action-value estimate for the chosen action
            q_values[action] = q_values[action] + (reward - q_values[action]) / (step + 1)

            internal_step += 1


