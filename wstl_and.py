import numpy as np
import matplotlib.pyplot as plt
#define the shape of the environment (11x11)
environment_rows = 6
environment_columns = 6
q_values = np.zeros((environment_rows, environment_columns,4))
#define actions
#numeric action codes: 0=up, 1=right, 2=down, 3=left
actions = ['right', 'down', 'left', 'up']

rewards = np.full((environment_rows,environment_columns), -100.)
aisles = {}
aisles[1] = [i for i in range(1,5)]
aisles[2] = [i for i in range(1,5)]
aisles[3] = [i for i in range(1,5)]
aisles[4] = [i for i in range(1,5)]

for row_index in range(1,5):
    for column_index in aisles[row_index]:
        rewards[row_index, column_index] = -1.
rewards[4,1]=-100.
rewards[4,2]=-100.
rewards[3,5]=-100.

#DEFINE HELPER FUNCTION

def is_terminal_state(current_row_index, current_column_index):
    if rewards[current_row_index, current_column_index] == -1.:
        return False
    else:
        return True

def get_starting_location():
    current_row_index = np.random.randint(environment_rows)
    current_column_index = np.random.randint(environment_columns)
    while is_terminal_state(current_row_index, current_column_index):
        current_row_index = np.random.randint(environment_rows)
        current_column_index = np.random.randint(environment_columns)
    return current_row_index, current_column_index


#define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
def get_next_action(current_row_index, current_column_index, epsilon):
    if np.random.random() > epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    else:
        return np.random.randint(4)

#define a function that will get the next location based on the chosen action
def get_next_location(current_row_index, current_column_index, action_index):
    i = current_row_index
    j = current_column_index
    if actions[action_index] =='up' and current_row_index > 0:
        i -=1
    elif actions[action_index] == 'right' and current_column_index < environment_columns-1:
        j +=1
    elif actions[action_index] == 'left' and current_column_index > 0:
        j -=1
    elif actions[action_index] == 'down' and current_row_index < environment_rows-1:
        i +=1
    return i, j

def satisfaction(step, current_row_index):
    if step >= 0 and current_row_index == condition1:
        sat1 = 1
    else:
        sat1 = -1
    if step >= 5 and current_row_index == condition2:
        sat2 = 1
    elif step >= 5 and current_row_index != condition2:
        sat2 = -1
    else:
        sat2 = 0
    if step >= 0 and current_row_index == condition3:
        sat3 = 1
    else:
        sat3 = -1
    return sat1, sat2, sat3

condition1=3
condition2=4
condition3=4
weights = [4,5,2]
def reward(current_row_index, sat1, sat2, sat3):
    rew = [0, 0, 0]
    rew[0] = ((current_row_index - condition1) ** 2) * sat1/25
    rew[1] = ((current_row_index - condition2) ** 2) * sat2/25
    rew[2] = ((current_row_index - condition3) ** 2) * sat3/25
    wts_s = sum(weights)
    sig = [np.sign(rew[i]) for i in range(3)]
    weighted_rew = min((((1/2 - (weights[i]/wts_s))*sig[i] + 1 / 2) * rew[i]) for i in range(3))
    return weighted_rew
#Train the agent using q-learning algorithm
#define training parameters
epsilon=0.20
discount_factor = 0.90
learning_rate = 0.95
#row_index, column_index = 0, 0
for episode in range(100000):
    print('episode', episode)
    row_index, column_index = get_starting_location()
    while not is_terminal_state(row_index, column_index):
        for t in range(9):
            action_index = get_next_action(row_index, column_index, epsilon)
            old_row_index, old_column_index = row_index, column_index
            c_row_index, c_column_index = get_next_location(row_index, column_index, action_index)
            sat1, sat2, sat3 = satisfaction(t, c_row_index)
            rew = reward(c_row_index,sat1,sat2,sat3)
            old_q_value = q_values[old_row_index, old_column_index, action_index]
            temporal_difference = rew + (discount_factor * np.max(q_values[c_row_index, c_column_index])) - old_q_value
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_values[old_row_index, old_column_index, action_index] = new_q_value
            row_index, column_index = c_row_index, c_column_index
            print(old_row_index, old_column_index, actions[action_index], row_index, column_index, new_q_value)
    #row_index, column_index = np.random.randint(6), np.random.randint(6)
    #print(q_values)
    #row_index, column_index = np.random.randint(6), 0


print('training complete')
print(q_values)
def get_shortest_path(start_row_index, start_column_index):
    shortest_path = [[2, 1]]
    row_index, column_index = start_row_index, start_column_index
    for t in range(7):
        action_index = get_next_action(row_index, column_index, 0)
        current_row_index, current_column_index = get_next_location(row_index, column_index, action_index)
        shortest_path.append([current_row_index, current_column_index])
        row_index, column_index = current_row_index, current_column_index
    return shortest_path

path = get_shortest_path(2,1)
print(path)
