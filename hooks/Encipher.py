class Encipher():
    deltas = []
    n = 2
    N_ks = 16
    def __init__(self) -> None:
        pass
    
    def get_delta(self):
        delt =  self.deltas[self.n]
        self.n -= 1
        return delt
encipher = Encipher()