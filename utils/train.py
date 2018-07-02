'''
Helper functions for training pytorch models.

'''

def train_step(model, loss_fn, optimizer, input_batch, label_batch):
    '''
    Performs one training step of the provided model.
    '''

    model.zero_grad()

    output = model(input_batch)
    # print(output)
    loss = loss_fn(output, label_batch)

    loss.backward()
    optimizer.step()

    return output, loss.item()
