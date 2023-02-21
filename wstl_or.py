import numpy as np
import matplotlib.pyplot as plt
#define the shape of the environment (11x11)
environment_rows = 6
environment_columns = 6
q_values = np.zeros((environment_rows, environment_columns,4))
#define actions
#numeric action codes: 0=up, 1=right, 2=down, 3=left
actions = ['up', 'right', 'down','left']
#DEFINE HELPER FUNCTION
#define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
def get_next_action(current_row_index, current_column_index, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    else:
        return np.random.randint(4)

#define a function that will get the next location based on the chosen action
def get_next_location(current_row_index, current_column_index, action_index):
    new_row_index = current_row_index
    new_column_index = current_column_index
    if actions[action_index] =='up' and current_row_index > 0:
        new_row_index -=1
    elif actions[action_index] == 'right' and current_column_index < environment_columns-1:
        new_column_index +=1
    elif actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -=1
    elif actions[action_index] == 'down' and current_row_index < environment_rows-1:
        current_row_index +=1
    return new_row_index, new_column_index

condition1=1
condition2=2
weights = [4, 1]
def reward(current_row_index):
    rew = [0, 0]
    if current_row_index <= condition1:
        rew[0] = (current_row_index - condition1) ** 2
    else:
        rew[0] = -(current_row_index - condition1) ** 2

    if current_row_index >= condition2:
        rew[1] = (current_row_index - condition2) ** 2
    else:
        rew[1] = -(current_row_index - condition2) ** 2
    wts_s = sum(weights)
    sig = [np.sign(rew[i]) for i in range(2)]
    weighted_rew = max(((-(1/2 - (weights[i]/wts_s))*sig[i] + 1 / 2) * rew[i]) for i in range(2))

    return weighted_rew

#Train the agent using q-learning algorithm
#define training parameters
epsilon=0.80
discount_factor = 0.90
learning_rate = 0.80
row_index, column_index = 4, 0
for episode in range(10000):
    print('episode', episode)
    for t in range(8):
        action_index = get_next_action(row_index, column_index, epsilon)
        old_row_index, old_column_index = row_index, column_index
        row_index, column_index = get_next_location(row_index, column_index, action_index)
        rew = reward(row_index)
        old_q_value = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = rew + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value
        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_row_index, old_column_index, action_index] = new_q_value
    row_index, column_index = 4, 0

print('training complete')
print(q_values)
def get_shortest_path(start_row_index, start_column_index):
    shortest_path = [[4, 0]]
    current_row_index, current_column_index = start_row_index, start_column_index
    for t in range(8):
        action_index = get_next_action(current_row_index, current_column_index, 1)
        current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
        shortest_path.append([current_row_index, current_column_index])
    return shortest_path

path = get_shortest_path(4,0)
print(path)
