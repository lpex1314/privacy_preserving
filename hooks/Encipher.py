class Encipher():
    deltas = []
    n = 2
    N_ks = 16
    def __init__(self) -> None:
        pass
    
    def get_delta(self):
        delt =  self.deltas[self.n]
        self.n -= 1
        if self.n == 0:
            self.n = 2
        return delt
encipher = Encipher()