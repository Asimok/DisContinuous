import json


def split_dataset(path, out_path, num):
    json_data = []
    i = 1
    with open(path) as f:
        new_data = f.readline()
        while new_data:
            if i > num:
                break
            json_data.append(json.loads(f.readline()))
            new_data = f.readline()
            i += 1
        # sava_data
    with open(out_path, 'w') as f:
        for tmp in json_data:
            f.write(json.dumps(tmp)+'\r')


train_path = '/Users/maqi/dataset/default/train.json'
train_out_path = '/Users/maqi/dataset/default/train-s.json'
split_dataset(train_path, train_out_path, 10)

dev_path = '/Users/maqi/dataset/default/dev.json'
dev_out_path = '/Users/maqi/dataset/default/dev-s.json'
split_dataset(dev_path, dev_out_path, 10)

test_path = '/Users/maqi/dataset/default/test.json'
test_out_path = '/Users/maqi/dataset/default/test-s.json'
split_dataset(test_path, test_out_path, 10)

print('ok')
