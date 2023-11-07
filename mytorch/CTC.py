import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):
        """
        
        Initialize instance variables

        Argument(s)
        -----------
        
        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        ext_symbols = [self.BLANK]
        for symbol in target:
            ext_symbols.append(symbol)
            ext_symbols.append(self.BLANK)

        N = len(ext_symbols)
        target_len = target.shape[0]
        # TODO: initialize skip_connect to be all zeros.

        # -------------------------------------------->
        skip_connect = np.zeros((2*target_len + 1,))
        for i in range(2,N-2):
            if ext_symbols[i-2] == self.BLANK:
                if ext_symbols[i-1] != ext_symbols[i-2]:
                    skip_connect[i+1] = 1
        # <---------------------------------------------

        ext_symbols = np.array(ext_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        return ext_symbols, skip_connect
        #raise NotImplementedError


    def get_forward_probs(self, logits, ext_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        S, T = len(ext_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        alpha[0][0] = logits[0][ext_symbols[0]]
        alpha[0][1] = logits[0][ext_symbols[1]]
            
        for t in range(1,T):
            alpha[t][0] = alpha[t-1][0]*logits[t][ext_symbols[0]]
            for i in range(1,S):
                alpha[t][i] = alpha[t-1][i-1] + alpha[t-1][i]
                if i>1 and ext_symbols[i]!=ext_symbols[i-2]:
                    alpha[t][i] += alpha[t-1][i-2]
                alpha[t][i] *= logits[t][ext_symbols[i]]
        # IMP: Remember to check for skipConnect when calculating alpha
        # <---------------------------------------------

        return alpha
        #raise NotImplementedError


    def get_backward_probs(self, logits, ext_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
        
        """

        S, T = len(ext_symbols), len(logits)
        beta = np.zeros(shape=(T, S))
        # beta_hat = np.zeros(shape=(T,S))

        beta[T-1][S-1] = logits[T-1][ext_symbols[S-1]]
        beta[T-1][S-2] = logits[T-1][ext_symbols[S-2]]
        # beta_hat[T][:S-2] = 0
        for t in reversed(range(T-1)):
            beta[t][S-1] = beta[t+1][S-1]*logits[t][ext_symbols[S-1]]
            for i in reversed(range(S-1)):
                beta[t][i] = beta[t+1][i] + beta[t+1][i+1]
                if i<=S-3 and ext_symbols[i] != ext_symbols[i+2]:
                    beta[t][i] += beta[t+1][i+2]
                beta[t][i] *= logits[t][ext_symbols[i]]
        for t in reversed(range(T)):
            for i in reversed(range(S)):
                beta[t][i] = beta[t][i]/logits[t][ext_symbols[i]]
        
        return beta
        
        # # return beta
        # raise NotImplementedError
        

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        T, S = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))
        
        # Compute normalization factor
        for t in range(T):
            sumgamma[t] = 0
            for i in range(S):
                gamma[t][i] = alpha[t][i]*beta[t][i]
                sumgamma[t] += gamma[t][i]
            for i in range(S):
                gamma[t][i] = gamma[t][i]/sumgamma[t]
        return gamma

class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.
        
        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior proabilites, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.ext_symbols = []

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO


            # Truncate the target to target length
            
            trunc_target = self.target[batch_itr][:self.target_lengths[batch_itr]]
            
            trunc_logits = self.logits[:self.input_lengths[batch_itr],batch_itr]
            
            ext_symbols, skip_connect = self.ctc.extend_target_with_blank(trunc_target)
            
            alpha = self.ctc.get_forward_probs(trunc_logits,ext_symbols,skip_connect)
            
            beta = self.ctc.get_backward_probs(trunc_logits,ext_symbols,skip_connect)
            
            gamma = self.ctc.get_posterior_probs(alpha,beta)

            T = gamma.shape[0]
            S = gamma.shape[1]

            # Compute expected divergence for each batch and store it in totalLoss
            for t in range(T):
                for s in range(S):
                    total_loss[batch_itr] -= np.log(trunc_logits[t][ext_symbols[s]])*gamma[t,s] 
 
        total_loss = np.sum(total_loss) / B
    
        return total_loss
        

    def backward(self):
        """
        
        CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative 
        w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(ext_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        for batch_itr in range(B):
           
            # Get the current batch's target sequence and input length
            trunc_target = self.target[batch_itr][:self.target_lengths[batch_itr]]

            
            trunc_logits = self.logits[:self.input_lengths[batch_itr],batch_itr]
            
            ext_symbols, skip_connect = self.ctc.extend_target_with_blank(trunc_target)
            
            alpha = self.ctc.get_forward_probs(trunc_logits,ext_symbols,skip_connect)
            
            beta = self.ctc.get_backward_probs(trunc_logits,ext_symbols,skip_connect)
            
            gamma = self.ctc.get_posterior_probs(alpha,beta)
            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            T = gamma.shape[0]
            
            S = gamma.shape[1]

            for t in range(T):
                for s in range(S):
                    dY[t,batch_itr,ext_symbols[s]] -= gamma[t,s]/trunc_logits[t,ext_symbols[s]]
        return dY