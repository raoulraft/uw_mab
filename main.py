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

    # Wait for the next reward to be written to the file before taking the next step
    while True:
        with open('rewards.csv') as rewards_file_check:
            rewards_reader_check = csv.reader(rewards_file_check)
            # next(rewards_reader_check)  # skip the header row
            row = next(rewards_reader_check)
            step = int(row[0])
            reward = float(row[1])
            # print("step and reward found in rewards.csv: {}, {}".format(step, reward))

        if step == internal_step:
            print("step and reward found in rewards.csv: {}, {}".format(step, reward))
            break

        if step < internal_step:
            # print("waiting for the reward to be written by AnPa (last_step < internal_step)")
            pass

        if step > internal_step:
            print(f"you messed up somewhere, since step found on csv ({step}) > internal_step ({internal_step})"
                  " (scripts are probably not coordinated on which step to start?)")
            exit(-1)

        time.sleep(0.1)

    # Update the action-value estimate for the chosen action
    q_values[action] = q_values[action] + (reward - q_values[action]) / (step + 1)

    internal_step += 1


