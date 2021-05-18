import torch
import torch.nn.functional as F
from torch.autograd import Variable
from codebase import torch_utils

class StructuredSelfAttention(torch.nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup
    """

    def __init__(self,
                 lstm_hid_dim,
                 d_a,
                 r,
                 max_len,
                 use_gpu = False):
        """
        Initializes parameters suggested in paper
 
        Args:
            batch_size  : {int} batch_size used for training
            lstm_hid_dim: {int} hidden dimension for lstm
            d_a         : {int} hidden dimension for the dense layer
            r           : {int} attention-hops or attention heads
            max_len     : {int} number of lstm timesteps
            emb_dim     : {int} embeddings dimension
            vocab_size  : {int} size of the vocabulary
            use_pretrained_embeddings: {bool} use or train your own embeddings
            embeddings  : {torch.FloatTensor} loaded pretrained embeddings
            type        : [0,1] 0-->binary_classification 1-->multiclass classification
            n_classes   : {int} number of classes
 
        Returns:
            self
 
        Raises:
            Exception
        """
        super(StructuredSelfAttention,self).__init__()
       
        #self.embeddings,emb_dim = self._load_embeddings(use_pretrained_embeddings,embeddings,vocab_size,emb_dim)
        #self.lstm = torch.nn.LSTM(emb_dim,lstm_hid_dim,1,batch_first=True)
        self.linear_first = torch.nn.Linear(lstm_hid_dim,d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a,r)
        self.linear_second.bias.data.fill_(0)
        #self.n_classes = n_classes
        #self.linear_final = torch.nn.Linear(lstm_hid_dim,self.n_classes)
        #self.batch_size = batch_size
        self.max_len = max_len
        self.lstm_hid_dim = lstm_hid_dim
        #self.hidden_state = self.init_hidden()
        self.r = r
        self.use_gpu = use_gpu
        #self.type = type

    """
    def _load_embeddings(self,use_pretrained_embeddings,embeddings,vocab_size,emb_dim):
        #Load the embeddings based on flag
       
        if use_pretrained_embeddings is True and embeddings is None:
            raise Exception("Send a pretrained word embedding as an argument")
           
        if not use_pretrained_embeddings and vocab_size is None:
            raise Exception("Vocab size cannot be empty")
   
        if not use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(vocab_size,emb_dim,padding_idx=0)
            
        elif use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
            word_embeddings.weight = torch.nn.Parameter(embeddings)
            emb_dim = embeddings.size(1)
            
        return word_embeddings,emb_dim
    """
        
    def softmax(self,input, axis=1):
        """
        Softmax applied to axis=n
 
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied
 
        Returns:
            softmaxed tensors
 
       
        """
 
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)
       
    """    
    def init_hidden(self):
        return (Variable(torch.zeros(1,self.batch_size,self.lstm_hid_dim)),Variable(torch.zeros(1,self.batch_size,self.lstm_hid_dim)))
    """
        
    def forward(self,input, use_reg=False, mask=None):
        #embeddings = self.embeddings(x)
        #outputs, self.hidden_state = self.lstm(embeddings.view(self.batch_size,self.max_len,-1),self.hidden_state)
        x = torch.tanh(self.linear_first(input))
        x = self.linear_second(x)       
        #x = self.softmax(x,1)
        x[~mask.unsqueeze(2).expand(mask.shape[0], mask.shape[1], x.shape[2])] = float('-inf')
        x = torch.softmax(x, dim=1)
        attention = x.transpose(1,2)
        sentence_embeddings = attention@input
        avg_sentence_embeddings = torch.sum(sentence_embeddings,1)/self.r

        if not use_reg:
            return avg_sentence_embeddings
        else:
            attT = attention.transpose(1,2)
            identity = torch_utils.gpu(torch.eye(attention.size(1)), self.use_gpu)
            identity = Variable(identity.unsqueeze(0).expand(input.shape[0],attention.size(1),attention.size(1)))
            penal = self.l2_matrix_norm(attention@attT - identity)


            return avg_sentence_embeddings, penal

	#Regularization
    def l2_matrix_norm(self,m):
        """
        Frobenius norm calculation
 
        Args:
           m: {Variable} ||AAT - I||
 
        Returns:
            regularized value
 
       
        """
        return torch.sum(torch.sum(m**2,1),1)**0.5#.type(torch.DoubleTensor)