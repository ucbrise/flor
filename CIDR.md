# Getting Started with FlorDB

Today I'm excited to introduce FlorDB, a versatile *hindsight logging database* that simplifies how we manage the AI and machine learning lifecycle.

Let me start by explaining what makes FlorDB unique. While there are many tools out there for managing ML workflows, FlorDB introduces something particularly useful: *hindsight logging*. Imagine you're many hours into training a model, and you suddenly realize you forgot to track an important metric. Traditionally, this would mean starting over from scratch. But with FlorDB, you can add those logging statements after the fact and efficiently replay your training with the new logging in place -- often in just seconds.

FlorDB is designed to integrate seamlessly with your existing workflow. Whether you're using Make for basic automation, Airflow for complex pipelines, MLFlow for experiment tracking, or Slurm for cluster management – FlorDB works alongside all of them.

What makes FlorDB particularly useful is its adaptability. It can serve as your:
- Git-aware logging library
- Checkpoint/Restore system for long-running Python tasks
- Model registry for version control
- Feature store for materializing results of featurization
- Label management solution for data annotation
- And more, adapting to your specific needs

## Installation

Getting started with FlorDB is straightforward. If you want the latest stable version, you can simply run:

```bash
pip install flordb
```

## Just Start Logging

Let me show you how simple it is to start using FlorDB. One of our core design principles is "low floor, high ceiling" – meaning it should be easy to get started, but capable enough for complex use cases.

Here's all you need to log your first message:

```python
import flor
flor.log("msg", "Hello world!")
```

When you run this, you'll see:
```
msg: Hello, World!
Changes committed successfully
```

And retrieving your logs? Just as simple. You can use a Flor Dataframe:

```python
import flor
flor.dataframe("msg")
```

This gives you a clean, organized view of all your logged messages. No need to set up a database schema, no complex configurations – just straightforward logging capabilities.

What's particularly useful about this approach is that you can start small, logging just the basics, and expand your logging as your needs grow. There's no upfront commitment to a complex infrastructure – FlorDB grows with your project's needs.

[Pause for transition]

Now that we've covered the basics, let me show you how FlorDB handles more complex scenarios, like tracking machine learning experiments...

## Logging Your Experiments

Now let's look at how FlorDB handles real machine learning workflows. While the basic logging we just saw is useful, FlorDB really shines when working with complex experiments that have multiple hyperparameters and metrics to track.

Let me show you how you can adapt your existing PyTorch training script to incorporate FlorDB logging. We'll break this down into three key parts: logging hyperparameters, managing model checkpoints, and tracking metrics.

First, let's look at how we handle hyperparameters:

```python
import flor
import torch

# Define and log hyperparameters
hidden_size = flor.arg("hidden", default=500)
batch_size = flor.arg("batch_size", 32)
learning_rate = flor.arg("lr", 1e-3)
```

Notice how we're using `flor.arg` here. This does two important things: it logs the parameter values, and it makes them configurable from the command line. This means you can easily run experiments with different parameters without changing your code:

```bash
python train.py --kwargs hidden=250 lr=5e-4
```

Next, let's look at the training loop. FlorDB provides a checkpointing system that works seamlessly with PyTorch:

```python
# Use FlorDB's checkpointing to manage model states
with flor.checkpointing(model=net, optimizer=optimizer):
    for epoch in flor.loop("epoch", range(num_epochs)):
        for data in flor.loop("step", trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Log the loss value for each step
            flor.log("loss", loss.item())

        # Evaluate the model on the test set
        eval(net, testloader)
```

Let me highlight a few important features here:
- The `flor.checkpointing` context manager handles saving and loading model states
- `flor.loop` helps track iteration progress
- `flor.log` captures metrics like loss values during training

To view all this logged information, you can use a Flor Dataframe just like before, but now with multiple columns:

```python
import flor
flor.dataframe("hidden", "batch_size", "lr", "loss")
```

This gives you a comprehensive view of your experiment, showing how different hyperparameters affect your model's performance.

[Pause for transition]

Now that we've covered basic experiment logging, let me show you one of FlorDB's most powerful features: hindsight logging...

## Hindsight Logging

Let me introduce you to one of FlorDB's most distinctive capabilities: hindsight logging - (which is) the ability to add logging statements after your experiments have run. It's particularly useful when you realize you need to track something you didn't think of initially. Let me walk you through a practical example.

First, let's use a sample repository to demonstrate this. You can get it by running:

```bash
git clone https://github.com/rlnsanz/ml_tutorial.git
cd ml_tutorial
make install
```

Let's start with our first training run:

```bash
python train.py
```

When we run this, we'll see output like:
```
Created and switched to new branch: flor.shadow
device: cuda
seed: 9288
hidden: 500
epochs: 5
batch_size: 32
lr: 0.001
print_every: 500
[Training progress output...]
accuracy: 90.9
correct: 9090
```

Now, let's run another experiment with different parameters:

```bash
python train.py --kwargs epochs=3 batch_size=64 lr=0.0005
```

At this point, we have two training runs in our database. We can view them easily:

```python
import flor
flor.dataframe("device", "seed", "epochs", "batch_size", "lr", "accuracy")
```

Here's where hindsight logging becomes valuable. Imagine you're analyzing these runs and realize you need to know what the gradient norms were during training. A naive solution is simply to retrain both models from scratch after adding the relevant logging statement. With FlorDB, you simply add the new logging statement to the latest version of your code and replay your previous runs efficiently.

Here's how we add gradient norm logging to our training script:

```python
flor.log("gradient_norm", 
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=float('inf')
    ).item()
)
```

To replay our previous runs with this new logging, we simply use:

```bash
python -m flor replay gradient_norm
```

FlorDB will analyze your previous runs and replay them efficiently, only executing the parts needed to capture the new information. When it's done, you can view the updated results:

```python
import flor
flor.dataframe("seed", "batch_size", "lr", "gradient_norm")
```

The power here is that we didn't have to modify our original experiments or rerun them from scratch. FlorDB handled all the complexity of propagating logging statements back in time and replaying the necessary parts of our training history.

[Pause for transition]

Now that we've covered the core features of FlorDB, let me show you how it fits into larger AI/ML applications...

## Building AI/ML Applications with FlorDB

Let me show you how FlorDB serves as more than just a logging system. In the real world, AI/ML applications need to manage complex pipelines spanning multiple components - from feature computation to model training to human feedback. Let me demonstrate this using our Document Parser application.

### FlorDB as a Feature Store

When processing PDF documents, we need to extract and store various features. Here's how FlorDB handles this:

```python
# featurize.py

for doc_name in flor.loop("document", os.listdir(...)):
    N = get_num_pages(doc_name)
    for page in flor.loop("page", range(N)):
        # text_src is 'OCR' or 'TXT'
        text_src, page_text = read_page(doc_name, page)
        flor.log("text_src", text_src)
        flor.log("page_text", page_text)

        # Run featurization
        headings, page_numbers = analyze_text(page_text)
        flor.log("headings", headings)
        flor.log("page_numbers", page_numbers)
```

What's important to note is that FlorDB automatically tracks:
- The source of each feature (OCR vs raw text)
- Text features like headings and page numbers
- The relationship between documents, pages, and their features
- Complete provenance of how features were computed

All this happens without needing a predefined schema or complex setup. We'll see this in action next.

### FlorDB as a Model Registry 

Now let me show you how FlorDB manages model training and versioning:

```python
# train.py

# Flor Dataframe for training data
labeled_data = flor.dataframe("first_page", "page_color")

# Define and track model parameters
hidden_size = flor.arg("hidden", default=500)
num_epochs = flor.arg("epochs", 5)
batch_size = flor.arg("batch_size", 32)
learning_rate = flor.arg("lr", 1e-3)

with flor.checkpointing(model=net, optimizer=optimizer):
    for epoch in flor.loop("epoch", range(num_epochs)):
        for data in flor.loop("step", trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            flor.log("loss", loss.item())
            optimizer.step()
        
        # Log evaluation metrics
        acc, recall = eval(net, testloader)
        flor.log("acc", acc)
        flor.log("recall", recall)
```

During inference, we can automatically select the best model:
XXX TODO I think there is code missing below to filter for best?

```python
# infer.py

# Query for best model based on metrics
best_model = flor.dataframe("acc", "recall")
```

### FlorDB for Feedback Loops

One of the most powerful aspects is how FlorDB handles human feedback. In our PDF Parser application, we have a Flask interface where experts can review and correct model predictions:

XXX TODO it's unclear from the code snippet how you are gettign human corrections and how they're distinguished from machine predictions.

```python
# app.py

@app.route("/save_colors", methods=["POST"])
def save_colors():
    colors = request.get_json().get("colors", [])
    pdf_name = pdf_names.pop()
    with flor.iteration("document", None, pdf_name):
        for i in flor.loop("page", range(len(colors))):
            # FlorDB saves changes to ground truth (i.e. feedback)
            flor.log("page_color", colors[i])
    flor.commit()
    return jsonify({"message": "Colors saved"}), 200
```

The key here is that FlorDB maintains complete provenance of both machine predictions and human corrections, making it easy to:
- Track which predictions were corrected
- Use corrections to improve model training
- Maintain data quality over time

XXX TODO The above asserts benefits that aren't really apparent from the demo.

```python
import flor
flor.dataframe("first_page", "page_color")
```

This gives you a complete view of the data, including both machine-generated and human-corrected labels.

### Putting it all together

Let's look at how all these pieces fit together in a real ML application. Here's our complete pipeline as defined in the Makefile:

```Makefile
process_pdfs: $(PDFS) pdf_demux.py
    @echo "Processing PDF files..."
    @python pdf_demux.py
    @touch process_pdfs

featurize: process_pdfs featurize.py
    @echo "Featurizing Data..."
    @python featurize.py
    @touch featurize

train: featurize hand_label train.py
    @echo "Training..."
    @python train.py

model.pth: train export_ckpt.py
    @echo "Generating model..."
    @python export_ckpt.py

infer: model.pth infer.py
    @echo "Inferencing..."
    @python infer.py
    @touch infer

hand_label: label_by_hand.py
    @echo "Labeling by hand"
    @python label_by_hand.py
    @touch hand_label

run: featurize infer
    @echo "Starting Flask..."
    @flask run
```

We've decided to manage dependencies and dataflow using Make, but you could just as easily use Airflow, Kubeflow, or any other workflow management system. FlorDB operates at the Python layer, and adapts to your existing infrastructure, making it easy to integrate into your AI/ML applications.

XXX TODO Walk the audience through how running "make foo" results in flor being invoked.

# Wrapping Up

So, that's FlorDB - a versatile logging database that brings together experiment tracking, model management, and human feedback in one cohesive system. Let me point you to some resources to learn more:

### Getting Started
To recap, the simplest way to get started is to install FlorDB via pip:
```bash
pip install flordb
```

### Resources
- Source code and example projects: https://github.com/ucbrise/flor
- Sample applications like the PDF Parser: https://github.com/rlnsanz/document_parser

### Getting Help
- For bug reports: Open an issue on the GitHub repository
- For academic collaborations or research questions, you can contact Rolando Garcia at rolando.garcia@asu.edu

### Join the Community
FlorDB is actively maintained and developed at ASU's School of Computing & Augmented Intelligence (SCAI), building on years of research at UC Berkeley's RISE Lab. We welcome contributions and feedback from the community.

If you're interested in learning more about the research behind FlorDB, check out our papers:
- "Flow with FlorDB: Incremental Context Maintenance for the Machine Learning Lifecycle" (CIDR 2025)
- "Hindsight Logging for Model Training" (VLDB Journal, 2021)

Thank you for your interest in FlorDB. We look forward to seeing what you'll build with it!
