import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):

        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size

        # Embedding vector
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)

        # LSTM
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first =True, dropout = 0.5)

        # Output fully connected layer
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

        # Initialize the weights
        self.init_weights()

    def init_weights(self):
        '''
        Method to initialize weights of the model with Xavier
        '''
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.embed.weight)
    
    def forward(self, features, captions):
        # Copy of captions
        captions = captions[:, :-1]

        # Embed the captions
        captions_embed = self.embed(captions)

        # Concat the extracted image features and captions
        inputs = torch.cat((features.unsqueeze(1), captions_embed), 1)

        # Run intputs through LSTM to obtain outputs
        outputs, _ = self.lstm(inputs)

        # Convert outputs to word predictions
        predictions = self.fc(outputs)

        return predictions


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # Initialize the output
        output_list = []
        
        # Counter to keep under max length
        count = 0

        # Keep track of token to search for the end of sentence token
        cur_token = None

        while count <= max_len and cur_token != 1:

            # LSTM
            lstm_output, states = self.lstm(inputs, states)

            # Linear layer
            output = self.fc(lstm_output)

            # Predicted next word 
            _, word = output.max(2)

            # Append token to output list
            cur_token = word.item()
            output_list.append(cur_token)

            # Current token is then the input for the next prediction
            inputs = self.embed(word)

            # Add to the count
            count += 1

        return output_list

        
        