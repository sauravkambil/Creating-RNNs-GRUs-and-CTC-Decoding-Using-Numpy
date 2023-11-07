import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        self.symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """
        symbol, seq_len, batch = y_probs.shape
        forward_path = []
        forward_prob = 1
        #since y_probs = symbols +1
        self.symbol_set = [''] + self.symbol_set
        last_index = 0
        for i in range(seq_len):
            index = int(np.argmax(y_probs[:, i, :], axis=0))
            forward_prob *= y_probs[index, i]
            if index != 0:
                #repeated error
                if bool(forward_path) == False or self.symbol_set[index] != forward_path[-1] or last_index == 0:
                    forward_path.append(self.symbol_set[index])
            last_index = index

        combined_forward_path = ''
        for i in forward_path:
            combined_forward_path += i

        forward_path = combined_forward_path
        return (forward_path, forward_prob)


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        self.symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """
        
        T = y_probs.shape[1]
        bestPath, FinalPathScore = None, None
        PathScore, BlankPathScore = dict(), dict()
        
        def InitializePaths(SymbolSet, y):
            InitialBlankPathScore, InitialPathScore = dict(), dict()
            path = ""
            InitialBlankPathScore[path] = y[0] 
            InitialPathsWithFinalBlank = {path}

            InitialPathsWithFinalSymbol = set()
            for i in range(len(SymbolSet)):
                InitialPathScore[SymbolSet[i]] = y[i+1]
                InitialPathsWithFinalSymbol.add(SymbolSet[i])
            
            return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore

        def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y):
            UpdatedPathsWithTerminalBlank = set()
            UpdatedBlankPathScore = dict()
            for path in PathsWithTerminalBlank:
                if path not in UpdatedPathsWithTerminalBlank:
                    UpdatedPathsWithTerminalBlank.add(path)
                    UpdatedBlankPathScore[path] = BlankPathScore[path]*y[0]

            for path in PathsWithTerminalSymbol:
                if path in UpdatedPathsWithTerminalBlank:
                    UpdatedBlankPathScore[path] += PathScore[path]* y[0]
                else:
                    UpdatedPathsWithTerminalBlank.add(path)
                    UpdatedBlankPathScore[path] = PathScore[path] * y[0]
                    
            return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore

        def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y):
            UpdatedPathsWithTerminalSymbol = set()
            UpdatedPathScore = dict()
            for path in PathsWithTerminalBlank:
                for i, c in enumerate(SymbolSet):
                    newpath = path + c
                    UpdatedPathsWithTerminalSymbol.add(newpath)
                    UpdatedPathScore[newpath] = BlankPathScore[path] * y[i+1]
            
            for path in PathsWithTerminalSymbol:
                for i, c in enumerate(SymbolSet):
                    if (c == path[-1]):
                        newpath = path 
                    else: 
                        newpath = path + c
                    if newpath in UpdatedPathsWithTerminalSymbol:
                        UpdatedPathScore[newpath] += PathScore[path] * y[i+1]
                    else:
                        UpdatedPathsWithTerminalSymbol.add(newpath)
                        UpdatedPathScore[newpath] = PathScore[path] * y[i+1]
            
            return UpdatedPathsWithTerminalSymbol, UpdatedPathScore
        
        def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
            PrunedBlankPathScore = dict()
            PrunedPathScore = dict()
            scorelist = []

            for p in PathsWithTerminalBlank:
                scorelist.append(BlankPathScore[p])
            
            for p in PathsWithTerminalSymbol:
                scorelist.append(PathScore[p])
          
            scorelist = sorted(scorelist)[::-1]
            if BeamWidth < len(scorelist):
                cutoff = scorelist[BeamWidth] 
            else: 
                cutoff = scorelist[-1]

            PrunedPathsWithTerminalBlank = set()
            for p in PathsWithTerminalBlank:
                if BlankPathScore[p] > cutoff :
                    PrunedPathsWithTerminalBlank.add(p)
                    PrunedBlankPathScore[p] = BlankPathScore[p]

            PrunedPathsWithTerminalSymbol = set()
            for p in PathsWithTerminalSymbol:
                if PathScore[p] > cutoff :
                    PrunedPathsWithTerminalSymbol.add(p)
                    PrunedPathScore[p] = PathScore[p]
            
            return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore


        def MergeIdenticalPaths(PathsWithTerminalBlank, BlankPathScore, PathsWithTerminalSymbol, PathScore):
            MergedPaths = PathsWithTerminalSymbol 
            FinalPathScore = PathScore
            for p in PathsWithTerminalBlank:
                if p in MergedPaths:
                    FinalPathScore[p] += BlankPathScore[p]
                else:
                    MergedPaths.add(p)
                    FinalPathScore[p] = BlankPathScore[p]
            
            return MergedPaths, FinalPathScore
        
        NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = InitializePaths(self.symbol_set, y_probs[:,0])

        for t in  range(1,T):
            PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore, self.beam_width)
            NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:,t])
            NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, self.symbol_set, y_probs[:,t])
       
        MergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore, NewPathsWithTerminalSymbol, NewPathScore)
        bestPath, _ = sorted(FinalPathScore.items(), key=lambda x: x[1])[-1]            
        
        return bestPath, FinalPathScore
