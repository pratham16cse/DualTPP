import numpy as np
import os

np.random.seed(42)
labels = np.arange(1, 11)
enc_len = 6
dec_len = 4

input_mark_data = np.random.randint(1, 11, size=(102000, 6))
input_time_data = np.random.randint(1, 11, size=(102000, 1)) + np.tile(np.arange(enc_len), [102000, 1])
output_mark_data = np.flip(input_mark_data[:, 2:], axis=1)
output_time_data = input_time_data[:, [-1]] + np.tile(np.arange(1, dec_len+1), [102000, 1])
print(input_time_data)
print(output_time_data)
mark_data = np.concatenate([input_mark_data, output_mark_data], axis=1)
time_data = np.concatenate([input_time_data, output_time_data], axis=1)

train_mark_input, train_mark_output = mark_data[:100000, :6], mark_data[:100000, 6:]
dev_mark_input, dev_mark_output = mark_data[100000:101000, :6], mark_data[100000:101000, 6:]
test_mark_input, test_mark_output = mark_data[101000:102000, :6], mark_data[101000:102000, 6:]

train_time_input, train_time_output = time_data[:100000, :6], time_data[:100000, 6:]
dev_time_input, dev_time_output = time_data[100000:101000, :6], time_data[100000:101000, 6:]
test_time_input, test_time_output = time_data[101000:102000, :6], time_data[101000:102000, 6:]
def get_diff(data):
    return data[:, 1:] - data[:, :-1]
train_time_output = get_diff(np.concatenate([train_time_input[:, -1:], train_time_output], axis=1))
dev_time_output = get_diff(np.concatenate([dev_time_input[:, -1:], dev_time_output], axis=1))
test_time_output = get_diff(np.concatenate([test_time_input[:, -1:], test_time_output], axis=1))


def write_to_file(fptr, sequences):
    for sequence in sequences:
        for s in sequence:
            f.write(str(s) + ' ')
        f.write('\n')

if not os.path.exists('synthetic'):
    os.mkdir('synthetic')

with open('synthetic/train.event.in', 'w') as f:
    write_to_file(f, train_mark_input.tolist())
with open('synthetic/dev.event.in', 'w') as f:
    write_to_file(f, dev_mark_input.tolist())
with open('synthetic/test.event.in', 'w') as f:
    write_to_file(f, test_mark_input.tolist())
with open('synthetic/train.event.out', 'w') as f:
    write_to_file(f, train_mark_output.tolist())
with open('synthetic/dev.event.out', 'w') as f:
    write_to_file(f, dev_mark_output.tolist())
with open('synthetic/test.event.out', 'w') as f:
    write_to_file(f, test_mark_output.tolist())
with open('synthetic/train.time.in', 'w') as f:
    write_to_file(f, train_time_input.tolist())
with open('synthetic/dev.time.in', 'w') as f:
    write_to_file(f, dev_time_input.tolist())
with open('synthetic/test.time.in', 'w') as f:
    write_to_file(f, test_time_input.tolist())
with open('synthetic/train.time.out', 'w') as f:
    write_to_file(f, train_time_output.tolist())
with open('synthetic/dev.time.out', 'w') as f:
    write_to_file(f, dev_time_output.tolist())
with open('synthetic/test.time.out', 'w') as f:
    write_to_file(f, test_time_output.tolist())

with open('synthetic/labels.in', 'w') as f:
    for lbl in labels:
        f.write(str(lbl) + '\n')
