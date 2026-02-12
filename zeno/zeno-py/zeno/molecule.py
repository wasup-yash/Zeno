"""
Pipeline composition (Molecules = collections of Atoms)
"""
from typing import List

class Molecule:
    """Compose multiple atomic operations into a pipeline"""
    
    def __init__(self, atoms: List):
        self.atoms = atoms
        self._fitted = False
    
    def fit(self, data):
        """Fit all atoms in sequence"""
        for atom in self.atoms:
            if hasattr(atom, 'fit'):
                atom.fit(data)
        self._fitted = True
        return self
    
    def transform(self, data):
        """Transform data through all atoms"""
        if not self._fitted:
            raise ValueError("Pipeline not fitted. Call .fit() first.")
        
        result = data
        for atom in self.atoms:
            if hasattr(atom, 'transform'):
                result = atom.transform(result)
        return result
    
    def fit_transform(self, data):
        """Fit and transform in one call"""
        return self.fit(data).transform(data)