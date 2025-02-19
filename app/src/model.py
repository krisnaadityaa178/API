# model.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pymongo import MongoClient  
from config import Config

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Add question prefix to help model distinguish question type
        question_prefix = "Pertanyaan: "
        context_prefix = "Konteks: "
        
        # Add special tokens and prefixes
        encoding = self.tokenizer(
            question_prefix + item['question'],
            context_prefix + item['context'],
            max_length=self.max_length,
            truncation='only_second',  # Only truncate context if needed
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=True,
            stride=128  # Add sliding window for long contexts
        )
        
        # Convert answer positions to token positions more accurately
        answer_start = item['context'].find(item['answer'])
        answer_end = answer_start + len(item['answer'])
        
        offset_mapping = encoding.pop('offset_mapping').squeeze()
        
        # Find token positions that contain the answer
        start_positions = []
        end_positions = []
        
        for idx, (start, end) in enumerate(offset_mapping):
            if start <= answer_start and end >= answer_start:
                start_positions.append(idx)
            if start <= answer_end and end >= answer_end:
                end_positions.append(idx)
        
        # Take the first valid positions
        if start_positions and end_positions:
            start_token = start_positions[0]
            end_token = end_positions[-1]
        else:
            # If answer not found, point to [CLS] token
            start_token = 0
            end_token = 0
            
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_positions': torch.tensor(start_token, dtype=torch.long),
            'end_positions': torch.tensor(end_token, dtype=torch.long)
        }

class UniversityChatbot:
    def __init__(self, model_name: str = Config.MODEL_NAME):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained(model_name).to(self.device)
        
        # Add special tokens for better context understanding
        special_tokens = {
            'additional_special_tokens': [
                '[NAMA]', '[/NAMA]', '[NPM]', '[/NPM]', '[PRODI]', '[/PRODI]', 
                '[FAKULTAS]', '[/FAKULTAS]', '[EMAIL]', '[/EMAIL]', '[ALAMAT]', '[/ALAMAT]'
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # MongoDB connection
        self.client = MongoClient(Config.MONGODB_URI)
        self.db = self.client[Config.DB_NAME]

    def _format_info(self, data: Dict[str, Any]) -> str:
        """Format informasi menjadi text yang mudah dibaca"""
        info = []
        fields = [('NAMA_LENGKAP', 'Nama Lengkap'), 
                    ('NPM', 'NPM'), 
                    ('PRODI', 'Prodi'), 
                    ('FAKULTAS', 'Fakultas'), 
                    ('EMAIL', 'Email'), 
                    ('ALAMAT', 'Alamat')]
        
        for field, display_name in fields:
            if field in data and data[field] and data[field] != "----":
                info.append(f"{display_name}: {data[field]}")
        
        return "\n".join(info)

    def _create_training_examples(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Enhanced training example creation with significantly more variations"""
        examples = []
        
        try:
            if 'NAMA_LENGKAP' in data and data['NAMA_LENGKAP']:
                nama = data['NAMA_LENGKAP']
                # Get first name for more natural questions
                first_name = nama.split()[0] if " " in nama else nama
                
                # Format context with special tokens
                context = (
                    f"[NAMA]{data.get('NAMA_LENGKAP', '')}[/NAMA] "
                    f"[NPM]{data.get('NPM', '')}[/NPM] "
                    f"[PRODI]{data.get('PRODI', '')}[/PRODI] "
                    f"[FAKULTAS]{data.get('FAKULTAS', '')}[/FAKULTAS] "
                    f"[EMAIL]{data.get('EMAIL', '')}[/EMAIL] "
                    f"[ALAMAT]{data.get('ALAMAT', '')}[/ALAMAT]"
                )
                
                # Comprehensive question variations
                qa_pairs = [
                    # Full information queries
                    {"question": f"tampilkan seluruh informasi mengenai {nama}", "answer": self._format_info(data)},
                    {"question": f"siapa {nama}?", "answer": self._format_info(data)},
                    {"question": f"berikan informasi lengkap tentang {nama}", "answer": self._format_info(data)},
                    {"question": f"ceritakan tentang mahasiswa bernama {nama}", "answer": self._format_info(data)},
                    
                    # NPM queries with variations
                    {"question": f"berapa NPM {nama}?", "answer": str(data.get('NPM', ''))},
                    {"question": f"NPM mahasiswa {nama}", "answer": str(data.get('NPM', ''))},
                    {"question": f"apa nomor {nama}?", "answer": str(data.get('NPM', ''))},
                    {"question": f"tolong beritahu NPM dari {nama}", "answer": str(data.get('NPM', ''))},
                    {"question": f"NPM {first_name}", "answer": str(data.get('NPM', ''))},
                    
                    # Program study variations
                    {"question": f"apa program studi {nama}?", "answer": str(data.get('PRODI', ''))},
                    {"question": f"jurusan {nama}", "answer": str(data.get('PRODI', ''))},
                    {"question": f"{nama} kuliah di jurusan apa?", "answer": str(data.get('PRODI', ''))},
                    {"question": f"program studi {first_name} apa?", "answer": str(data.get('PRODI', ''))},
                    {"question": f"di jurusan apa {nama} kuliah?", "answer": str(data.get('PRODI', ''))},
                    
                    # Faculty queries
                    {"question": f"{nama} kuliah dimana?", "answer": str(data.get('FAKULTAS', ''))},
                    {"question": f"fakultas {nama}", "answer": str(data.get('FAKULTAS', ''))},
                    {"question": f"di fakultas mana {nama} belajar?", "answer": str(data.get('FAKULTAS', ''))},
                    {"question": f"{first_name} di fakultas apa?", "answer": str(data.get('FAKULTAS', ''))},
                    
                    # Contact information
                    {"question": f"bagaimana cara menghubungi {nama}?", "answer": str(data.get('EMAIL', ''))},
                    {"question": f"apa email {nama}?", "answer": str(data.get('EMAIL', ''))},
                    {"question": f"email dari {first_name}", "answer": str(data.get('EMAIL', ''))},
                    {"question": f"kontak {nama}", "answer": str(data.get('EMAIL', ''))},
                    
                    # Address information
                    {"question": f"dimana {nama} tinggal?", "answer": str(data.get('ALAMAT', ''))},
                    {"question": f"alamat {nama}", "answer": str(data.get('ALAMAT', ''))},
                    {"question": f"berikan alamat {first_name}", "answer": str(data.get('ALAMAT', ''))},
                ]
                
                # Add context to all examples
                examples.extend([
                    {"question": qa["question"], "context": context, "answer": qa["answer"]}
                    for qa in qa_pairs
                    if qa["answer"] and qa["answer"] != "----"
                ])
                
        except Exception as e:
            print(f"Error creating training examples: {str(e)}")
        
        return examples

    def train(self, epochs: int = Config.EPOCHS, batch_size: int = Config.BATCH_SIZE, resume=False):
        """Enhanced training procedure with better optimization and monitoring and resuming capability"""
        print("Starting enhanced training procedure...")
        try:
            # Get starting epoch
            start_epoch = 0
            best_val_loss = float('inf')
            training_history = []
            
            if resume:
                # Load training history if exists
                history_path = os.path.join(Config.MODEL_PATH, 'training_history.json')
                if os.path.exists(history_path):
                    with open(history_path, 'r') as f:
                        training_history = json.load(f)
                        
                    if training_history:
                        start_epoch = training_history[-1]['epoch']
                        best_val_loss = min([entry.get('val_loss', float('inf')) for entry in training_history])
                        print(f"Resuming from epoch {start_epoch} with best validation loss: {best_val_loss:.4f}")
            
            # Prepare and augment training data
            training_data = self.prepare_training_data()
            
            # Split data with stratification
            train_data, val_data = train_test_split(
                training_data, 
                test_size=Config.TRAIN_TEST_SPLIT,
                random_state=Config.RANDOM_SEED,
                shuffle=True
            )
            
            # Create datasets with improved tokenization
            train_dataset = QADataset(train_data, self.tokenizer, Config.MAX_LENGTH)
            val_dataset = QADataset(val_data, self.tokenizer, Config.MAX_LENGTH)
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=torch.cuda.is_available()
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=torch.cuda.is_available()
            )
            
            # Initialize optimizer
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY
            )
            
            # Initialize scheduler
            num_training_steps = len(train_loader) * (epochs - start_epoch)
            num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            
            # Training loop with improved monitoring
            patience_counter = 0
            
            for epoch in range(start_epoch, epochs):
                # Training phase
                self.model.train()
                train_loss = 0
                
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
                
                # For checkpointing within epoch (every N batches)
                batch_checkpointing_freq = 100  # Save checkpoint every 100 batches
                total_batches_processed = 0
                
                for batch_idx, batch in enumerate(progress_bar):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Clear gradients
                    optimizer.zero_grad()
                    
                    # Forward pass with gradient accumulation
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Backward pass with gradient clipping
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    
                    train_loss += loss.item()
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                    total_batches_processed += 1
                    
                    # Save checkpoint within epoch (every N batches)
                    if total_batches_processed % batch_checkpointing_freq == 0:
                        checkpoint_path = os.path.join(
                            Config.MODEL_PATH, 
                            f'checkpoint_model_epoch_{epoch+1}_batch_{total_batches_processed}'
                        )
                        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                        self.model.save_pretrained(checkpoint_path)
                        self.tokenizer.save_pretrained(checkpoint_path)
                        
                        # Save current point in training loop
                        with open(os.path.join(checkpoint_path, 'training_state.json'), 'w') as f:
                            json.dump({
                                'epoch': epoch,
                                'batch_idx': batch_idx,
                                'total_batches_processed': total_batches_processed,
                                'current_train_loss': train_loss / (batch_idx + 1)
                            }, f, indent=4)
                        
                        print(f"\nIntermediate checkpoint saved at {checkpoint_path}")
                
                avg_train_loss = train_loss / len(train_loader)
                
                # Validation phase
                self.model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validating"):
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        outputs = self.model(**batch)
                        val_loss += outputs.loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                
                # Record history
                epoch_info = {
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'learning_rate': scheduler.get_last_lr()[0]
                }
                training_history.append(epoch_info)
                
                print(f"\nEpoch {epoch + 1}")
                print(f"Average Training Loss: {avg_train_loss:.4f}")
                print(f"Validation Loss: {avg_val_loss:.4f}")
                print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    model_path = os.path.join(Config.MODEL_PATH, f'best_model_epoch_{epoch+1}')
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    
                    self.model.save_pretrained(model_path)
                    self.tokenizer.save_pretrained(model_path)
                    
                    # Save training metrics with the model
                    metrics = {
                        'best_epoch': epoch + 1,
                        'best_val_loss': best_val_loss,
                        'train_loss': avg_train_loss,
                        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                    }
                    
                    with open(os.path.join(model_path, 'metrics.json'), 'w') as f:
                        json.dump(metrics, f, indent=4)
                    
                    print(f"Saved best model with validation loss: {avg_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                        print("\nEarly stopping triggered!")
                        break
                
                # Save checkpoint after each epoch
                checkpoint_path = os.path.join(Config.MODEL_PATH, f'checkpoint_model_epoch_{epoch+1}')
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                self.model.save_pretrained(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)
                
                # Save training history after each epoch
                history_path = os.path.join(Config.MODEL_PATH, 'training_history.json')
                with open(history_path, 'w') as f:
                    json.dump(training_history, f, indent=4)
            
            print("\nTraining completed!")
            print(f"Best validation loss: {best_val_loss:.4f}")
            
            # Save the final model
            self.save_model(
                is_final=True, 
                metadata={
                    'best_val_loss': best_val_loss,
                    'epochs_trained': epoch + 1,
                    'training_completed': True
                }
            )
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def prepare_training_data(self) -> List[Dict[str, Any]]:
        """Prepare training data from MongoDB documents"""
        print("Preparing training data...")
        training_examples = []
        
        try:
            # Fetch all documents from MongoDB
            all_docs = list(self.db.mahasiswas.find({}))
            
            # Create training examples for each document
            for doc in tqdm(all_docs, desc="Processing documents"):
                examples = self._create_training_examples(doc)
                training_examples.extend(examples)
            
            print(f"Created {len(training_examples)} training examples")
        except Exception as e:
            print(f"Error preparing training data: {str(e)}")
        
        return training_examples

    def save_model(self, is_final=False, metadata=None):
        """
        Menyimpan model dengan format yang lebih terstruktur
        
        Args:
            is_final (bool): True jika ini adalah model final setelah training
            metadata (dict): Metadata tambahan untuk disimpan dengan model
        """
        try:
            # Buat direktori jika belum ada
            os.makedirs(Config.MODEL_PATH, exist_ok=True)
            
            # Tentukan nama model berdasarkan tipe
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if is_final:
                model_name = 'final_model'
            else:
                model_name = f'checkpoint_model_{timestamp}'
            
            model_path = os.path.join(Config.MODEL_PATH, model_name)
            
            # Simpan metadata
            if metadata is None:
                metadata = {}
            metadata.update({
                'saved_at': timestamp,
                'is_final': is_final,
                'model_type': 'BertForQuestionAnswering',
                'tokenizer_vocab_size': len(self.tokenizer),
                'device_used': str(self.device)
            })
            
            # Simpan model dan tokenizer
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            
            # Simpan metadata
            with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print(f"Model berhasil disimpan di {model_path}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_type='final', model_path=None):
        """
        Memuat model dengan opsi yang lebih fleksibel
        
        Args:
            model_type (str): Tipe model yang akan dimuat
                            'final' - memuat final_model
                            'latest' - memuat checkpoint terbaru
                            'best' - memuat model dengan performa terbaik
            model_path (str): Path spesifik untuk memuat model
        """
        try:
            if model_path and os.path.exists(model_path):
                print(f"Loading model dari {model_path}")
            else:
                model_path = None
                
                if model_type == 'final':
                    # Coba memuat final model
                    final_path = os.path.join(Config.MODEL_PATH, 'final_model')
                    if os.path.exists(final_path):
                        model_path = final_path
                    else:
                        print("Final model tidak ditemukan, mencoba checkpoint terbaru...")
                        model_type = 'latest'
                
                if model_type == 'latest':
                    # Cari semua checkpoint
                    checkpoints = [d for d in os.listdir(Config.MODEL_PATH) 
                                    if d.startswith('checkpoint_model_')]
                    
                    if not checkpoints:
                        raise FileNotFoundError("Tidak ada checkpoint model yang ditemukan")
                    
                    # Ambil checkpoint terbaru berdasarkan timestamp
                    latest_checkpoint = max(checkpoints)
                    model_path = os.path.join(Config.MODEL_PATH, latest_checkpoint)
                
                elif model_type == 'best':
                    # Cari semua checkpoint yang ada best_model
                    best_models = [d for d in os.listdir(Config.MODEL_PATH) 
                                if d.startswith('best_model_epoch_')]
                    
                    if not best_models:
                        raise FileNotFoundError("Tidak ada model terbaik yang ditemukan")
                    
                    # Ambil model dengan validation loss terendah
                    best_val_loss = float('inf')
                    best_model_path = None
                    
                    for model_dir in best_models:
                        metrics_path = os.path.join(Config.MODEL_PATH, model_dir, 'metrics.json')
                        if os.path.exists(metrics_path):
                            with open(metrics_path, 'r') as f:
                                metrics = json.load(f)
                            
                            if metrics.get('best_val_loss', float('inf')) < best_val_loss:
                                best_val_loss = metrics.get('best_val_loss')
                                best_model_path = os.path.join(Config.MODEL_PATH, model_dir)
                    
                    if best_model_path:
                        model_path = best_model_path
                    else:
                        raise FileNotFoundError("Tidak dapat menemukan model terbaik dengan data metrik")
            
            if model_path is None:
                raise FileNotFoundError(f"Tidak dapat menemukan model dengan tipe {model_type}")
            
            print(f"Loading model dari {model_path}")
            
            # Validasi metadata sebelum loading
            metadata_path = os.path.join(model_path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"Model metadata: {metadata}")
            
            # Load model dan tokenizer
            self.model = BertForQuestionAnswering.from_pretrained(model_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            print("Model berhasil dimuat")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def _extract_name_from_question(self, question: str) -> Optional[str]:
        """Extract name from question using multiple patterns"""
        patterns = [
            r"mengenai\s+([^?]+)",           # tampilkan seluruh informasi mengenai [NAMA]
            r"NPM\s+([^?]+)[\?]?",           # berapa NPM [NAMA]?
            r"apa (?:program studi|jurusan)\s+([^?]+)[\?]?",  # apa program studi/jurusan [NAMA]?
            r"siapa\s+([^?]+)[\?]?",         # siapa [NAMA]?
            r"([^?]+) kuliah dimana[\?]?",   # [NAMA] kuliah dimana?
            r"cara menghubungi\s+([^?]+)[\?]?",  # cara menghubungi [NAMA]?
            r"email\s+([^?]+)[\?]?",         # email [NAMA]?
            r"alamat\s+([^?]+)[\?]?",        # alamat [NAMA]?
            r"fakultas\s+([^?]+)[\?]?",      # fakultas [NAMA]?
            r"nomor\s+([^?]+)[\?]?",         # nomor [NAMA]?
            r"jurusan\s+([^?]+)[\?]?",       # jurusan [NAMA]?
        ]
        
        name = None
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                break
        
        # If no pattern matched, try extracting last 2-3 words as a potential name
        if not name and len(question.split()) >= 2:
            words = question.split()
            potential_name = ' '.join(words[-2:])  # Try last two words as name
            
            # Check if potential_name exists in database
            doc = self.db.mahasiswas.find_one({"NAMA_LENGKAP": {"$regex": potential_name, "$options": "i"}})
            if doc:
                name = potential_name
        
        return name

    def answer_question(self, question: str) -> str:
        """Enhanced answer_question with better name extraction and fallback logic"""
        try:
            # Extract name from question
            name = self._extract_name_from_question(question)
            
            if not name:
                return "Maaf, saya tidak dapat mengidentifikasi nama mahasiswa. Mohon sertakan nama lengkap dalam pertanyaan."
            
            # Find relevant document from MongoDB
            doc = self.db.mahasiswas.find_one({"NAMA_LENGKAP": {"$regex": name, "$options": "i"}})
            
            if not doc:
                return f"Maaf, saya tidak dapat menemukan informasi untuk mahasiswa dengan nama '{name}'."
            
            # Add question prefixes to help the model understand the query type
            question_prefix = "Pertanyaan: "
            context_prefix = "Konteks: "
            
            # Create context string from document with clearer structure
            context = context_prefix + (
                f"[NAMA]{doc.get('NAMA_LENGKAP', '')}[/NAMA] "
                f"[NPM]{doc.get('NPM', '')}[/NPM] "
                f"[PRODI]{doc.get('PRODI', '')}[/PRODI] "
                f"[FAKULTAS]{doc.get('FAKULTAS', '')}[/FAKULTAS] "
                f"[EMAIL]{doc.get('EMAIL', '')}[/EMAIL] "
                f"[ALAMAT]{doc.get('ALAMAT', '')}[/ALAMAT]"
            )
            
            # Add the prefix to help model identify question type
            question_with_prefix = question_prefix + question
            
            # Tokenize question and context together
            inputs = self.tokenizer(
                question_with_prefix,
                context,
                max_length=Config.MAX_LENGTH,
                padding="max_length",
                truncation="only_second",
                return_tensors="pt"
            ).to(self.device)
            
            # Analyze the question to determine what type of information is requested
            question_lower = question.lower()
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get the most likely answer span
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits)
            
            # Convert token indices to text
            answer_tokens = inputs["input_ids"][0][answer_start:answer_end + 1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            # Clean up the answer
            answer = answer.strip()
            
            # Calculate confidence score for the answer
            start_scores = F.softmax(outputs.start_logits, dim=1)
            end_scores = F.softmax(outputs.end_logits, dim=1)
            score = start_scores[0][answer_start].item() * end_scores[0][answer_end].item()
            
            # Apply more effective fallback logic with scoring threshold
            if not answer or len(answer) < 3 or score < 0.1:
                # Identify the type of information being requested and return appropriate field
                if any(term in question_lower for term in ["npm", "nomor"]):
                    return f"NPM dari {doc.get('NAMA_LENGKAP')} adalah {doc.get('NPM', 'tidak tersedia')}"
                
                elif any(term in question_lower for term in ["prodi", "jurusan", "program studi"]):
                    return f"Program studi dari {doc.get('NAMA_LENGKAP')} adalah {doc.get('PRODI', 'tidak tersedia')}"
                
                elif any(term in question_lower for term in ["fakultas", "kuliah dimana"]):
                    return f"{doc.get('NAMA_LENGKAP')} kuliah di Fakultas {doc.get('FAKULTAS', 'tidak tersedia')}"
                
                elif any(term in question_lower for term in ["email", "kontak", "menghubungi"]):
                    return f"Email dari {doc.get('NAMA_LENGKAP')} adalah {doc.get('EMAIL', 'tidak tersedia')}"
                
                elif any(term in question_lower for term in ["alamat", "tinggal"]):
                    return f"Alamat dari {doc.get('NAMA_LENGKAP')} adalah {doc.get('ALAMAT', 'tidak tersedia')}"
                
                elif any(term in question_lower for term in ["informasi", "seluruh", "siapa"]):
                    return self._format_info(doc)
                
                else:
                    # Generic fallback: return the most likely field
                    return self._format_info(doc)
            
            return answer
        
        except Exception as e:
            print(f"Error answering question: {str(e)}")
            return "Maaf, terjadi kesalahan dalam memproses pertanyaan."

    def evaluate_model(self, test_questions: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate model on a list of test questions"""
        correct = 0
        total = len(test_questions)
        results = []
        
        for item in test_questions:
            question = item['question']
            expected_answer = item['expected_answer']
            
            # Get model's answer
            answer = self.answer_question(question)
            
            # Simple exact match evaluation
            is_correct = expected_answer.lower() in answer.lower()
            
            # Track results
            results.append({
                'question': question,
                'expected': expected_answer,
                'predicted': answer,
                'correct': is_correct
            })
            
            if is_correct:
                correct += 1
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0
        
        # Save evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(Config.MODEL_PATH, f'evaluation_results_{timestamp}.json')
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump({
                'accuracy': accuracy,
                'total_questions': total,
                'correct_answers': correct,
                'results': results
            }, f, indent=4)
        
        return {
            'accuracy': accuracy,
            'total': total,
            'correct': correct
        }

    def run_interactive_mode(self):
        """Run an interactive mode for testing the model"""
        print("\nInteractive QA Mode - Type 'exit' to quit")
        print("=========================================")
        
        while True:
            question = input("\nMasukkan pertanyaan: ")
            
            if question.lower() == 'exit':
                print("Terima kasih telah menggunakan chatbot!")
                break
            
            answer = self.answer_question(question)
            print(f"Jawaban: {answer}")