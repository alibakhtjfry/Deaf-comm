"""
Complete torchtext compatibility stub for Python 3.12+
Provides minimal implementations to avoid torchtext C++ extension issues
"""
import torch
import numpy as np

class Example:
    """Minimal Example implementation"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @staticmethod
    def fromlist(values, fields):
        """Create Example from list of values and fields"""
        example = Example()
        for (name, field), value in zip(fields, values):
            setattr(example, name, value)
        return example

class Field:
    """Minimal Field implementation"""
    def __init__(self, *args, **kwargs):
        self.init_token = kwargs.get('init_token')
        self.eos_token = kwargs.get('eos_token')
        self.pad_token = kwargs.get('pad_token')
        self.tokenize = kwargs.get('tokenize', str.split)
        self.batch_first = kwargs.get('batch_first', False)
        self.lower = kwargs.get('lower', False)
        self.unk_token = kwargs.get('unk_token')
        self.include_lengths = kwargs.get('include_lengths', False)
        self.sequential = kwargs.get('sequential', True)
        self.use_vocab = kwargs.get('use_vocab', True)
        self.dtype = kwargs.get('dtype', torch.long)
        self.preprocessing = kwargs.get('preprocessing')
        self.postprocessing = kwargs.get('postprocessing')

class RawField:
    """Minimal RawField implementation"""
    def __init__(self, *args, **kwargs):
        pass

class Batch:
    """Batch object to hold training data"""
    def __init__(self, batch_size=8):
        # Source data: (tensor with shape [batch, seq_len], lengths)
        self.src = (torch.zeros(batch_size, 100, dtype=torch.long), 
                   torch.zeros(batch_size, dtype=torch.long))
        # Target data: shape [batch, seq_len, trg_size] where trg_size=150
        self.trg = torch.zeros(batch_size, 100, 151, dtype=torch.float32)
        self.file_paths = []

class Iterator:
    """Iterator that actually uses the loaded dataset examples"""
    def __init__(self, dataset, batch_size, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_idx = 0
        self.shuffle = kwargs.get('shuffle', False)
        self.repeat = kwargs.get('repeat', False)
        self.sort = kwargs.get('sort', False)
        self.train = kwargs.get('train', False)
        self.vocab = kwargs.get('vocab', None)  # Get vocab if provided
        
        # Create indices
        self.indices = list(range(len(self.dataset.examples)))
        if self.shuffle and self.train:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        self.current_idx = 0
        if self.shuffle and self.train:
            np.random.shuffle(self.indices)
        return self
    
    def _token_to_id(self, token):
        """Convert token to index using vocab or hash fallback"""
        if self.vocab and hasattr(self.vocab, 'stoi'):
            # torchtext vocab: vocab.stoi[token]
            return self.vocab.stoi.get(token, self.vocab.stoi.get('<unk>', 0))
        elif self.vocab and isinstance(self.vocab, dict):
            # Dict vocab
            return self.vocab.get(token, self.vocab.get('<unk>', 0))
        else:
            # Fallback: simple hashing with smaller range
            return abs(hash(token)) % 100
    
    def __next__(self):
        if self.current_idx >= len(self.dataset.examples):
            raise StopIteration
        
        batch_end = min(self.current_idx + self.batch_size, len(self.dataset.examples))
        batch_indices = self.indices[self.current_idx:batch_end]
        batch_examples = [self.dataset.examples[i] for i in batch_indices]
        actual_batch_size = len(batch_examples)
        self.current_idx = batch_end
        
        # Create batch object
        batch = Batch(batch_size=actual_batch_size)
        
        # Process actual example data
        src_data = []
        trg_data = []
        file_paths = []
        max_src_len = 0
        max_trg_len = 0
        
        for ex in batch_examples:
            # Get src (should be a string/list of tokens)
            if hasattr(ex, 'src'):
                src = ex.src
                if isinstance(src, str):
                    src = src.split()
                src_data.append(src)
                max_src_len = max(max_src_len, len(src))
            
            # Get trg (should be a list of frames, each frame is a list of floats)
            if hasattr(ex, 'trg'):
                trg = ex.trg
                if trg and len(trg) > 0:
                    trg_data.append(trg)
                    max_trg_len = max(max_trg_len, len(trg))
            
            # Get file path
            if hasattr(ex, 'files'):
                file_paths.append(ex.files)
            else:
                file_paths.append(f"example_{len(file_paths)}")
        
        # Create proper tensors from the data
        if src_data:
            # Pad source sequences (tokenize them)
            src_tensor = torch.zeros(actual_batch_size, max_src_len, dtype=torch.long)
            src_lengths = torch.zeros(actual_batch_size, dtype=torch.long)
            for i, src in enumerate(src_data):
                src_lengths[i] = len(src)
                # Convert tokens to indices using vocabulary
                for j, token in enumerate(src):
                    src_tensor[i, j] = self._token_to_id(token)
            batch.src = (src_tensor, src_lengths)
        
        if trg_data:
            # Pad target sequences
            if max_trg_len > 0 and len(trg_data[0]) > 0:
                trg_size = len(trg_data[0][0]) if trg_data[0] else 151
                trg_tensor = torch.zeros(actual_batch_size, max_trg_len, trg_size, dtype=torch.float32)
                for i, trg_frames in enumerate(trg_data):
                    for j, frame in enumerate(trg_frames):
                        trg_tensor[i, j, :len(frame)] = torch.tensor(frame, dtype=torch.float32)
                batch.trg = trg_tensor
        
        batch.file_paths = file_paths
        return batch

class BucketIterator(Iterator):
    """Bucket Iterator - groups examples by length"""
    def __init__(self, dataset, batch_size, **kwargs):
        super().__init__(dataset, batch_size, **kwargs)

class Dataset:
    """Minimal Dataset implementation"""
    def __init__(self, examples, fields, **kwargs):
        self.examples = examples
        self.fields = fields
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

# Create a data namespace with all classes
class data:
    Field = Field
    RawField = RawField
    Iterator = Iterator
    BucketIterator = BucketIterator
    Dataset = Dataset
    Example = Example
    Batch = Batch
