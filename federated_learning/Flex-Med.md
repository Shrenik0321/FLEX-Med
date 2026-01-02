# FLEX-Med: Federated Learning for Medical Image Classification

## 📋 Table of Contents
1. [Overview](#overview)
2. [What Problem Does This Solve?](#what-problem-does-this-solve)
3. [Architecture Overview](#architecture-overview)
4. [Code Structure](#code-structure)
5. [Detailed Component Breakdown](#detailed-component-breakdown)
6. [Execution Flow](#execution-flow)
7. [Key Concepts Explained](#key-concepts-explained)
8. [Example Round Walkthrough](#example-round-walkthrough)
9. [Configuration Guide](#configuration-guide)

---

## 🎯 Overview

**FLEX-Med** is a Federated Learning system that trains AI models to detect leukemia from blood cell images **without hospitals sharing private patient data**. 

It implements **FedMD (Federated Model Distillation)**, a technique where:
- Multiple hospitals collaborate on AI training
- Each hospital keeps patient data private
- Knowledge is shared through "soft predictions" on public images
- Different hospitals can use different AI architectures

---

## 🏥 What Problem Does This Solve?

### Traditional AI Training Problem
```
Hospital A has 100 patient images
Hospital B has 150 patient images
Hospital C has 200 patient images

❌ Privacy laws prevent sharing patient data
❌ Each hospital trains separately = 3 weak models
```

### FedMD Solution
```
✅ No patient data leaves hospitals
✅ Hospitals share knowledge through "consensus"
✅ All hospitals benefit from collective knowledge
✅ Result: 1 strong collaborative model per hospital
```

### Real-World Scenario
```
🏥 Hospital A: Uses ResNet-18, has ALL-IDB2 dataset (small)
🏥 Hospital B: Uses MobileNetV2, has CNMC dataset (medium)
🏥 Hospital C: Uses DenseNet-121, has NO data (Free Rider)

After FedMD:
- Hospital A's model improves beyond its small dataset
- Hospital B's model learns from A's data distribution
- Hospital C gets a working model despite having no data!
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         FEDMD SYSTEM                             │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────┐
    │              CENTRAL SERVER                      │
    │  - Aggregates logits from all clients           │
    │  - Computes consensus (average predictions)     │
    │  - Coordinates training rounds                  │
    └────────┬─────────────────────────────┬───────────┘
             │                             │
    Sends    │                             │  Receives
    Consensus│                             │  Logits
             │                             │
    ┌────────▼─────────┐  ┌────────▼─────────┐  ┌────────▼─────────┐
    │   CLIENT 0       │  │   CLIENT 1       │  │   CLIENT 2       │
    │   (Hospital A)   │  │   (Hospital B)   │  │   (Hospital C)   │
    │                  │  │                  │  │                  │
    │  🏥 ResNet-18    │  │  🏥 MobileNetV2  │  │  🏥 DenseNet-121 │
    │  📊 ALL-IDB2     │  │  📊 CNMC         │  │  📊 No Data      │
    │  🔒 Private      │  │  🔒 Private      │  │  🚫 Free Rider   │
    └──────────────────┘  └──────────────────┘  └──────────────────┘
             │                      │                      │
             └──────────────────────┴──────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  PUBLIC DATASET     │
                    │  (Common Reference) │
                    │  🖼️ 200-300 images  │
                    │  📂 Mixed from both │
                    └─────────────────────┘
```

---

## 📁 Code Structure

```
flex-med/
│
├── server.py           # Server orchestration
├── client.py           # Client training logic
├── task.py             # Models, data loaders, FedMD strategy
│
└── Data Structure:
    /content/drive/MyDrive/College/
    ├── Datasets/fed_data/
    │   ├── public_anchor/       # Shared reference images
    │   │   ├── all/             # Leukemia images
    │   │   └── hem/             # Healthy images
    │   ├── client_allidb/       # Hospital A's private data
    │   │   ├── all/
    │   │   └── hem/
    │   └── client_cnmc/         # Hospital B's private data
    │       ├── all/
    │       └── hem/
    └── models/
        ├── model_client_0.pt    # Saved state for Client 0
        ├── model_client_1.pt    # Saved state for Client 1
        └── model_client_2.pt    # Saved state for Client 2
```

---

## 🔍 Detailed Component Breakdown

### 1️⃣ **server.py** - The Orchestrator

```python
@app.main()
def main(grid: Grid, context: Context) -> None:
```

**Purpose:** Coordinates the entire federated learning process

**Step-by-Step Breakdown:**

#### Step 1: Load Configuration
```python
num_rounds: int = context.run_config["num-server-rounds"]  # e.g., 10 rounds
lr: float = context.run_config["lr"]                       # e.g., 0.01
```
- Defines how many training rounds to run
- Sets the learning rate for all clients

#### Step 2: Initialize Zero Consensus
```python
public_loader = load_public_dataset(batch_size=1)
num_samples = len(public_loader.dataset)  # e.g., 250 images
num_classes = 2                           # Binary: Healthy vs Leukemia
zero_consensus = np.zeros((num_samples, num_classes), dtype=np.float32)
```

**What is this matrix?**
```
Shape: (250, 2)
Example:
[[0.0, 0.0],   ← Image 1: No predictions yet
 [0.0, 0.0],   ← Image 2: No predictions yet
 ...
 [0.0, 0.0]]   ← Image 250: No predictions yet
```

**Why zeros?** Round 1 starts fresh - no client has trained yet.

#### Step 3: Start Federated Training
```python
strategy = FedMDStrategy()
result = strategy.start(
    grid=grid,
    initial_arrays=ArrayRecord([zero_consensus]),
    train_config=ConfigRecord({"lr": lr}),
    num_rounds=num_rounds,
)
```
- Creates the FedMD coordination strategy
- Kicks off the training loop for N rounds

#### Step 4: Save Final Knowledge
```python
final_logits = result.arrays["0"].numpy()
np.save("final_consensus.npy", final_logits)
```
- After all rounds complete, saves the "universal knowledge"
- This represents what all hospitals collectively learned

---

### 2️⃣ **task.py** - The Brain (FedMDStrategy)

This file contains three critical components:

#### A. Model Definitions

```python
def get_resnet():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # 2 classes
    return model
```

**Why different models?**
- Simulates real hospitals with different infrastructure
- ResNet: More complex, needs more compute
- MobileNet: Lighter, runs on mobile devices
- DenseNet: Different architecture paradigm

#### B. Data Loaders

**Public Dataset (Critical for FedMD):**
```python
def load_public_dataset(batch_size=64):
    dataset = datasets.ImageFolder(root=PUBLIC_PATH, transform=COMMON_TRANSFORM)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # ← MUST be False!
    return loader
```

**Why `shuffle=False`?**
```
Client 0 sees: [img_1, img_2, img_3, ...]
Client 1 sees: [img_1, img_2, img_3, ...]  ← Same order!
Client 2 sees: [img_1, img_2, img_3, ...]

When averaging logits, they must align:
consensus[0] = average(client0_pred[0], client1_pred[0], client2_pred[0])
                        ↑ All for img_1
```

**Private Dataset:**
```python
def load_private_dataset(partition_id, num_partitions, batch_size=32):
    if partition_id == 0:
        data_path = CLIENT_0_PATH  # ALL-IDB2
    elif partition_id == 1:
        data_path = CLIENT_1_PATH  # CNMC
    elif partition_id == 2:
        return None, None          # Free Rider has no data
```

#### C. FedMD Strategy Methods

**Method 1: `configure_train()` - Start of Each Round**

```python
def configure_train(self, server_round, arrays, config, grid):
    """Send consensus to all clients to begin training"""
    
    for node_id in grid.get_node_ids():
        msg = Message(
            metadata=Metadata(..., group_id=str(server_round)),
            content=RecordDict({
                "arrays": arrays,   # ← Consensus logits from previous round
                "config": config    # ← Training hyperparameters
            })
        )
        messages.append(msg)
    
    return messages
```

**What gets sent:**
```python
# Round 1:
arrays = [[0.0, 0.0], [0.0, 0.0], ...]  # Zero consensus

# Round 2:
arrays = [[0.3, 0.7], [0.8, 0.2], ...]  # Real consensus from Round 1

# Round 3:
arrays = [[0.25, 0.75], [0.85, 0.15], ...]  # Updated consensus from Round 2
```

**Method 2: `aggregate_train()` - End of Each Round**

```python
def aggregate_train(self, server_round, results, **kwargs):
    """Receive logits from all clients and create consensus"""
    
    logits_list = []
    for msg in results:
        client_logits = msg.content["arrays"]["0"].numpy()
        logits_list.append(client_logits)
    
    # Average all predictions
    consensus_logits = np.mean(logits_list, axis=0)
    
    return ArrayRecord([consensus_logits]), metrics
```

**Example Aggregation:**
```python
# Client 0 predicts for Image 1:
client_0_logits[0] = [0.2, 0.8]  # 20% healthy, 80% leukemia

# Client 1 predicts for Image 1:
client_1_logits[0] = [0.4, 0.6]  # 40% healthy, 60% leukemia

# Client 2 predicts for Image 1:
client_2_logits[0] = [0.3, 0.7]  # 30% healthy, 70% leukemia

# Consensus:
consensus[0] = (0.2 + 0.4 + 0.3) / 3, (0.8 + 0.6 + 0.7) / 3
             = [0.3, 0.7]  # 30% healthy, 70% leukemia (collective wisdom)
```

**Method 3: `aggregate_evaluate()` - Performance Tracking**

```python
def aggregate_evaluate(self, server_round, results, **kwargs):
    """Aggregate evaluation metrics from all clients"""
    
    total_loss = 0.0
    total_acc = 0.0
    total_examples = 0
    
    for msg in results:
        metrics = msg.content.get("metrics", {})
        eval_loss = metrics.get("eval_loss", 0.0)
        eval_acc = metrics.get("eval_acc", 0.0)
        num_examples = metrics.get("num-examples", 0)
        
        # Weighted average by dataset size
        total_loss += eval_loss * num_examples
        total_acc += eval_acc * num_examples
        total_examples += num_examples
    
    avg_loss = total_loss / total_examples
    avg_acc = total_acc / total_examples
    
    return {"loss": avg_loss, "metrics": {"eval_acc": avg_acc}}
```

---

### 3️⃣ **client.py** - The Worker

```python
@app.train()
def train(msg: Message, context: Context):
```

Each client goes through **3 phases** per round:

#### **Phase 1: Knowledge Distillation** 🧠

```python
# Extract consensus from server
consensus_logits = msg.content["arrays"]["0"].numpy()

# Only distill if consensus exists (not Round 1)
if np.any(consensus_logits != 0):
    public_loader = load_public_dataset(batch_size=32)
    
    distill_loss = distill_knowledge(
        model=model,
        public_loader=public_loader,
        consensus_logits=consensus_logits,
        device=device,
        epochs=1,
        lr=0.001,
        temperature=2.0
    )
```

**What happens in `distill_knowledge()`?**

```python
def distill_knowledge(model, public_loader, consensus_logits, device, epochs, lr, temperature):
    """Teach the model to mimic consensus predictions"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    consensus_tensor = torch.from_numpy(consensus_logits).float()
    
    idx = 0
    for epoch in range(epochs):
        for images, _ in public_loader:
            images = images.to(device)
            batch_size = images.size(0)
            
            # Get consensus for this batch
            batch_consensus = consensus_tensor[idx:idx+batch_size].to(device)
            
            # Model's current predictions
            student_logits = model(images)
            
            # KL Divergence: How different are we from consensus?
            loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(batch_consensus / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)
            
            # Update model to be closer to consensus
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            idx += batch_size
        
        idx = 0  # Reset for next epoch
    
    return avg_loss
```

**Visual Example:**
```
Public Image 1:
├─ Consensus says: [0.3, 0.7] (30% healthy, 70% leukemia)
├─ My model says:  [0.6, 0.4] (60% healthy, 40% leukemia)
└─ Loss is HIGH → Update model to move closer to [0.3, 0.7]

After distillation:
└─ My model now says: [0.35, 0.65] (closer to consensus!)
```

**Why Temperature?**
```python
# Temperature = 1.0 (Hard)
softmax([2.0, 6.0]) = [0.02, 0.98]  # Very confident

# Temperature = 2.0 (Soft)
softmax([2.0, 6.0] / 2.0) = [0.12, 0.88]  # Less confident, easier to learn from
```

#### **Phase 2: Private Training** 🏥

```python
trainloader, _ = load_private_dataset(partition_id, num_partitions, batch_size=32)

if trainloader is not None:
    train_loss = train_fn(
        model=model,
        trainloader=trainloader,
        epochs=context.run_config["local-epochs"],  # e.g., 5 epochs
        lr=msg.content["config"]["lr"],
        device=device
    )
else:
    print("Free-Rider: No private data, skipping")
```

**Standard supervised learning:**
```python
def train(model, trainloader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    model.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return avg_loss
```

**What's different from Phase 1?**
```
Phase 1 (Distillation):
- Uses PUBLIC images
- Learns from CONSENSUS (soft labels)
- Goal: Align with other clients

Phase 2 (Private Training):
- Uses PRIVATE patient images
- Learns from TRUE labels (hard labels)
- Goal: Specialize on local data
```

#### **Phase 3: Generate Public Logits** 📤

```python
# Save model state for next round
torch.save(model.state_dict(), get_model_path(partition_id))

# Generate predictions on public dataset
public_loader = load_public_dataset(batch_size=32)
public_logits = get_public_logits(model, public_loader, device)

# Send back to server
return Message(content=RecordDict({
    "arrays": ArrayRecord([public_logits]),
    "metrics": MetricRecord({
        "train_loss": train_loss,
        "distill_loss": distill_loss,
        "num-examples": dataset_len
    })
}))
```

**What are these logits?**
```python
def get_public_logits(model, public_loader, device):
    """Generate predictions on all public images"""
    model.eval()
    
    all_logits = []
    with torch.no_grad():
        for images, _ in public_loader:
            images = images.to(device)
            outputs = model(images)  # Raw logits (not probabilities)
            all_logits.append(outputs.cpu().numpy())
    
    return np.concatenate(all_logits)
```

**Example output:**
```python
public_logits.shape = (250, 2)  # 250 images, 2 classes

# Client 0's predictions after this round:
[[0.5, 1.2],   # Image 1: Slightly favors leukemia
 [-0.3, 0.8],  # Image 2: Clearly leukemia
 [1.5, -0.2],  # Image 3: Clearly healthy
 ...]
```

---

## 🔄 Execution Flow

### **Complete System Flow**

```
┌────────────────────────────────────────────────────────────────┐
│                    INITIALIZATION                               │
└────────────────────────────────────────────────────────────────┘

[Server] Load config (num_rounds=10, lr=0.01)
[Server] Create zero consensus matrix (250 x 2)
[Server] Initialize FedMDStrategy
[Server] Call strategy.start()

┌────────────────────────────────────────────────────────────────┐
│                       ROUND 1                                   │
└────────────────────────────────────────────────────────────────┘

[Server] configure_train(round=1, arrays=zero_consensus)
    │
    ├─► [Client 0] Receives zero consensus
    │       ├─ Phase 1: Skip distillation (consensus is zeros)
    │       ├─ Phase 2: Train ResNet on ALL-IDB2 (5 epochs)
    │       ├─ Phase 3: Generate logits_0 on public dataset
    │       └─ Send logits_0 back to server
    │
    ├─► [Client 1] Receives zero consensus
    │       ├─ Phase 1: Skip distillation
    │       ├─ Phase 2: Train MobileNet on CNMC (5 epochs)
    │       ├─ Phase 3: Generate logits_1 on public dataset
    │       └─ Send logits_1 back to server
    │
    └─► [Client 2] Receives zero consensus
            ├─ Phase 1: Skip distillation
            ├─ Phase 2: No training (Free Rider, no data)
            ├─ Phase 3: Generate logits_2 on public dataset (random)
            └─ Send logits_2 back to server

[Server] aggregate_train(round=1, results=[logits_0, logits_1, logits_2])
    │
    └─► consensus_1 = mean([logits_0, logits_1, logits_2])
        Example: consensus_1[0] = [0.3, 0.7] ← First real consensus!

┌────────────────────────────────────────────────────────────────┐
│                       ROUND 2                                   │
└────────────────────────────────────────────────────────────────┘

[Server] configure_train(round=2, arrays=consensus_1)
    │
    ├─► [Client 0] Receives consensus_1
    │       ├─ Phase 1: Distill from consensus_1 ✅ NOW ACTIVE
    │       │           (Update ResNet to mimic [0.3, 0.7], [0.8, 0.2], ...)
    │       ├─ Phase 2: Train on ALL-IDB2 (refine after distillation)
    │       ├─ Phase 3: Generate new logits_0
    │       └─ Send logits_0 back
    │
    ├─► [Client 1] Receives consensus_1
    │       ├─ Phase 1: Distill from consensus_1 ✅
    │       ├─ Phase 2: Train on CNMC
    │       ├─ Phase 3: Generate new logits_1
    │       └─ Send logits_1 back
    │
    └─► [Client 2] Receives consensus_1
            ├─ Phase 1: Distill from consensus_1 ✅
            │           (DenseNet learns from others despite no data!)
            ├─ Phase 2: No training (still Free Rider)
            ├─ Phase 3: Generate new logits_2 (improved now!)
            └─ Send logits_2 back

[Server] aggregate_train(round=2, results=[logits_0, logits_1, logits_2])
    │
    └─► consensus_2 = mean([logits_0, logits_1, logits_2])
        Example: consensus_2[0] = [0.28, 0.72] ← Refined consensus

┌────────────────────────────────────────────────────────────────┐
│                    ROUNDS 3-10                                  │
└────────────────────────────────────────────────────────────────┘

[Same pattern repeats]
- Each round: Distillation → Private Training → Generate Logits
- Consensus gets more refined each round
- Client 2 keeps learning from consensus despite no private data

┌────────────────────────────────────────────────────────────────┐
│                      FINALIZATION                               │
└────────────────────────────────────────────────────────────────┘

[Server] All rounds complete
[Server] Save final_consensus.npy (the "universal knowledge")
[Server] Print metrics and statistics
```

---

## 🧠 Key Concepts Explained

### 1. **What are Logits?**

```python
# Raw model output (before softmax)
logits = [2.5, 0.3]

# Convert to probabilities
probs = softmax(logits) = [0.90, 0.10]  # 90% class 0, 10% class 1
```

**Why use logits instead of probabilities?**
- Logits preserve more information
- Easier to average across clients
- Better for mathematical operations in distillation

### 2. **Knowledge Distillation (KD)**

**Traditional Training:**
```
Image → Model → Hard Label
🩸     → CNN   → "This is Leukemia" (100% confident)
```

**Distillation Training:**
```
Image → Model → Soft Labels
🩸     → CNN   → "70% Leukemia, 30% Healthy" (captures uncertainty)
```

**Why it works:**
- Soft labels contain more information than hard labels
- Model learns "why" something is classified a certain way
- Captures nuances: "This looks like leukemia BUT has some healthy features"

### 3. **Consensus in FedMD**

**Think of it as democratic voting:**

```
Image of blood cell:

Doctor A (Client 0): "I'm 80% sure it's leukemia"
Doctor B (Client 1): "I'm 60% sure it's leukemia"
Doctor C (Client 2): "I'm 70% sure it's leukemia"

Consensus: "We're (80+60+70)/3 = 70% sure it's leukemia"

Next round:
Each doctor learns from this consensus and updates their opinion
```

### 4. **Why Free Rider Can Learn**

```
Round 1:
Client 2 (Free Rider) has random model → produces random logits

Round 2:
Client 2 receives consensus from Clients 0 & 1
├─ Distills this knowledge
└─ Now has a semi-trained model (learned from others!)

Round 3:
Client 2's logits improve further
├─ Contributes better predictions to consensus
└─ Helps other clients too!
```

### 5. **Temperature in Distillation**

```python
logits = [2.0, 6.0]

# Temperature = 1.0 (Hard)
softmax([2.0, 6.0] / 1.0) = [0.02, 0.98]
└─ Very confident: "98% sure it's class 1"

# Temperature = 2.0 (Soft)
softmax([2.0, 6.0] / 2.0) = [0.12, 0.88]
└─ Less confident: "88% sure it's class 1, but 12% chance of class 0"

# Temperature = 5.0 (Very Soft)
softmax([2.0, 6.0] / 5.0) = [0.27, 0.73]
└─ Much less confident: More room for learning
```

**Higher temperature = Softer distribution = Easier to learn from**

---

## 📊 Example Round Walkthrough

Let's trace **one image** through **Round 2**:

### Setup
```
Public Image #42: A blood cell image (true label: Leukemia)

After Round 1:
- Client 0 logits: [0.2, 1.8] → softmax = [0.20, 0.80]
- Client 1 logits: [0.4, 1.4] → softmax = [0.31, 0.69]
- Client 2 logits: [0.5, 0.9] → softmax = [0.38, 0.62] (random, untrained)

Consensus from Round 1:
consensus[42] = mean([[0.2, 1.8], [0.4, 1.4], [0.5, 0.9]]) = [0.37, 1.37]
```

### Round 2 Begins

#### Client 0 (ResNet):

**Phase 1: Distillation**
```python
# Load Image #42 from public dataset
image_42 = public_loader[42]

# Current model prediction
current_pred = model(image_42) = [0.3, 1.6]  # Different from last round

# Consensus says
consensus[42] = [0.37, 1.37]

# Compute KL divergence loss
loss = kl_div(current_pred, consensus[42]) = 0.08

# Update model to be closer to consensus
# After update: new_pred = [0.34, 1.48]  ← Moved closer!
```

**Phase 2: Private Training**
```python
# Train on ALL-IDB2 dataset
for epoch in range(5):
    for private_image, true_label in trainloader:
        loss = cross_entropy(model(private_image), true_label)
        # Update model based on hospital's private data
```

**Phase 3: Generate Logits**
```python
# Predict on Image #42 again (after both phases)
final_pred = model(image_42) = [0.15, 1.95]
# Send [0.15, 1.95] back to server
```

#### Client 1 (MobileNet):

**Phase 1: Distillation**
```python
current_pred = [0.5, 1.2]
consensus[42] = [0.37, 1.37]
# After distillation: [0.42, 1.31]
```

**Phase 2: Private Training** (on CNMC)
**Phase 3: Generate Logits**
```python
final_pred = [0.25, 1.75]  # Send to server
```

#### Client 2 (DenseNet - Free Rider):

**Phase 1: Distillation**
```python
current_pred = [0.5, 0.9]  # Was random in Round 1
consensus[42] = [0.37, 1.37]
# After distillation: [0.40, 1.20]  ← Learned from others!
```

**Phase 2: No Private Training** (skipped)

**Phase 3: Generate Logits**
```python
final_pred = [0.40, 1.20]  # Improved! (was [0.5, 0.9])
```

### Server Aggregates

```python
new_consensus[42] = mean([
    [0.15, 1.95],  # Client 0
    [0.25, 1.75],  # Client 1
    [0.40, 1.20]   # Client 2 (improved!)
]) = [0.27, 1.63]

# Compare to Round 1: [0.37, 1.37]
# → Consensus improved! More confident about leukemia (1.63 > 1.37)
```

### Round 3 Preview

This `[0.27, 1.63]` consensus will be sent back to all clients, and the cycle continues...

---

## ⚙️ Configuration Guide

### Required Setup

```python
# In your run configuration (YAML or dict):
{
    "num-server-rounds": 10,      # Number of federated rounds
    "local-epochs": 5,             # Epochs per client per round
    "lr": 0.01,                    # Learning rate
    "num-partitions": 3            # Number of clients
}
```

### Data Directory Structure

```
/content/drive/MyDrive/College/Datasets/fed_data/
├── public_anchor/
│   ├── all/               # Leukemia samples
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── hem/               # Healthy samples
│       ├── img001.jpg
│       └── ...
├── client_allidb/         # Hospital A's private data
│   ├── all/
│   └── hem/
└── client_cnmc/           # Hospital B's private data
    ├── all/
    └── hem/
```

### Model Checkpoints

```
/content/drive/MyDrive/College/models/
├── model_client_0.pt      # Automatically saved after each round
├── model_client_1.pt
└── model_client_2.pt
```

---

## 🎓 Learning Outcomes

After understanding this codebase, you should know:

1. ✅ How federated learning preserves privacy
2. ✅ How knowledge distillation transfers information
3. ✅ How consensus aggregation works
4. ✅ Why FedMD allows heterogeneous models
5. ✅ How free riders can benefit from collaboration
6. ✅ The difference between hard and soft labels
7. ✅ The role of public datasets in federated learning

---

## 🚀 Running the Code

```bash
# Start the simulation
flower-simulation server.py client.py --config config.yaml

# Expected output:
# [Server] Round 1: Configuring Training
# [Client 0] Round 1: No consensus yet. Skipping distillation.
# [Client 0] Training on private data...
# [Client 1] Training on private data...
# [Client 2] Free-Rider (No Private Data). Skipping.
# [Server] Aggregating Logits
# [Server] Round 2: Configuring Training
# [Client 0] Consensus received. Distilling...
# ...
```

---

## 📚 Additional Resources

- **FedMD Paper:** "FedMD: Heterogenous Federated Learning via Model Distillation"
- **Knowledge Distillation:** "Distilling the Knowledge in a Neural Network" by Hinton et al.
- **Flower Framework:** https://flower.ai/docs/

---

## 🤝 Contributing

To extend this codebase:

1. **Add new client:** Modify `load_model_for_client()` and `load_private_dataset()`
2. **Change aggregation:** Edit `aggregate_train()` in FedMDStrategy
3. **Add evaluation metrics:** Extend `aggregate_evaluate()`
4. **Experiment with temperature:** Adjust in `distill_knowledge()`

---

**Made with ❤️ for Federated Medical AI Research**
