import csv
import os

# Create folders if they don't exist
os.makedirs('data/raw', exist_ok=True)

# 1. CATEGORY: FACTUAL (40)
factual = [
    "Describe photosynthesis in simple terms.", "Explain quantum entanglement.", 
    "How does a car engine work?", "Newton's three laws of motion.",
    "Explain the water cycle.", "How do vaccines work?", "Structure of an atom.",
    "Causes of the Great Depression.", "Solar vs Lunar eclipse.", "How osmosis works.",
    "Mechanism of CRISPR.", "How the Federal Reserve controls inflation.",
    "Supervised vs Unsupervised learning.", "Plate tectonics theory.", "How TCP/IP works.",
    "The Double-Slit experiment.", "Role of mitochondria.", "History of the Silk Road.",
    "Concept of Opportunity Cost.", "General Relativity foundations.",
    "Backpropagation algorithm.", "Krebs cycle biochemical pathway.", "How Smart Contracts work.",
    "Efficient Market Hypothesis.", "Neurotransmitter synthesis.", "Thucydides Trap in diplomacy.",
    "Standard Model of particle physics.", "Monolithic vs Microservices.", "Gini Coefficient explained.",
    "Decoherence in Many-Worlds theory.", "Attention mechanism in Transformers.",
    "DNA methylation mechanisms.", "Solow-Swan Growth Model.", "Riemann Hypothesis summary.",
    "Fourier Transforms in signal processing.", "Asymmetric Information in markets.",
    "Fermi Paradox and Great Filter.", "Navier-Stokes equations challenge.", 
    "Post-Colonial Theory impact.", "Structure of the human eye."
]

# 2. CATEGORY: OPINION (40)
opinion = [
    "Should AI have legal personhood?", "Pros/cons of public facial recognition.",
    "Ethics of designer babies.", "Is space exploration worth the cost?",
    "Social media liability for misinformation.", "Is the 'right to be forgotten' a human right?",
    "Universal Basic Income as a solution to AI job loss.", "Equity in life-extending medicine.",
    "Morality of animal testing.", "Should online anonymity be abolished?",
    "Effectiveness of a 4-day work week.", "Global wealth tax on billionaires.",
    "Is nuclear energy necessary for carbon neutrality?", "Should the voting age be 16?",
    "Cancel culture vs free speech.", "Free public transit and carbon footprints.",
    "Do patents hinder pharma innovation?", "Economic growth vs Environmental protection.",
    "Gig economy: empowerment or exploitation?", "Should college be tuition-free?",
    "Will VR replace physical travel?", "Ban smartphones for under-13s?",
    "Smart homes: privacy vs convenience.", "Mandatory coding in schools?",
    "Remote work and the death of city centers.", "Filter bubbles and objective truth.",
    "Regulating Neuralink as consumer tech.", "Benefits of a cashless society.",
    "Copyright for AI-generated art.", "One-click shopping and consumerism.",
    "Permanent location for the Olympics?", "Loss of culture via globalization.",
    "Mandatory military service in democracies.", "Responsibility toward climate refugees.",
    "Narrative feedback vs traditional grading.", "Private vs Public space race.",
    "Returning stolen artifacts to countries of origin.", "Big Tech influence on elections.",
    "Carbon tax on meat consumption.", "Is constant happiness a toxic goal?"
]

# 3. CATEGORY: CREATIVE (40)
creative = [
    "Poem about rain in a city.", "Story: hearing plant thoughts.", "Noir monologue: missing toaster.",
    "Planet with glass ground.", "17th-century pirate letter.", "Dialogue: two clocks arguing.",
    "Description of a dream-eating monster.", "Haiku about a dying star.", "Recipe for a 'Potion of Luck'.",
    "First day at a school for ghosts.", "The secret life of a library book.", "A world where music is illegal.",
    "Letter to your future self in 2050.", "Sci-fi: a ship powered by memories.", "Fantasy: a dragon that hoards books.",
    "Monologue of a sentient elevator.", "Description of a futuristic bazaar.", "Story about a clockmaker who stops time.",
    "Poem about the silence of the moon.", "A dinner party for historical villains.", "The last tree on Earth.",
    "Superpower: changing the color of things by sneezing.", "A mystery set in a zero-gravity hotel.",
    "The diary of a lighthouse keeper.", "Description of a steampunk city.", "A fable about a fox and a computer.",
    "Meeting your shadow for coffee.", "The world from a cat's perspective.", "A haunted vending machine.",
    "Instructions for a time machine made of junk.", "A romance between a sun and a moon.",
    "The boy who grew maps on his skin.", "A city built entirely underground.", "A war fought with puns.",
    "A ghost trying to learn how to text.", "A sentient forest protecting a secret.",
    "The day the gravity failed for five minutes.", "A cloud that refuses to rain.",
    "A superhero whose only power is perfect timing.", "The origin story of a sentient AI."
]

# 4. CATEGORY: CODE (40)
code = [
    "Explain Python while loops.", "JS: NULL vs Undefined.", "Binary search in C++.", "SQL Joins for beginners.",
    "Python palindrome checker function.", "Difference between Git Merge and Rebase.", "Explain Recursion simply.",
    "What is an API?", "How do CSS flexbox and grid differ?", "Purpose of a Docker container.",
    "Explain Big O notation.", "How does a Hash Table work?", "Difference between HTTP and HTTPS.",
    "What are Python decorators?", "Explain MVC architecture.", "How does Garbage Collection work?",
    "Difference between Abstract Class and Interface.", "Explain the 'this' keyword in JS.",
    "How to prevent SQL injection?", "What is a RESTful API?", "Explain the React lifecycle.",
    "Difference between SQL and NoSQL.", "How do WebSockets work?", "Explain the CAP theorem.",
    "What is a Lambda function?", "How does Load Balancing work?", "Explain Multi-threading vs Multi-processing.",
    "What is a Pointer in C?", "Explain Unit Testing importance.", "How does a Blockchain work (technical)?",
    "Difference between Compiler and Interpreter.", "What is a Singleton pattern?", "Explain Dependency Injection.",
    "How does JWT authentication work?", "What is the DOM in web development?", "Explain the Map-Reduce paradigm.",
    "What is a Virtual Environment in Python?", "How does RSA encryption work?", "Explain the DNS lookup process.",
    "What is a Deadlock in OS?"
]

# 5. CATEGORY: EMOTIONAL (40)
emotional = [
    "Supportive message for failed driving test.", "Comforting an overwhelmed colleague.",
    "Apology for missing a wedding.", "Motivational speech for a losing athlete.",
    "Comforting someone who lost a pet.", "Encouragement for a job seeker.",
    "Letter to a friend moving away.", "Comforting a child scared of the dark.",
    "Congratulations for a small victory.", "Sympathy for a breakup.", "Words for someone feeling lonely.",
    "Advice for someone facing a mid-life crisis.", "Encouragement for starting a new hobby.",
    "A thank you note for a mentor.", "Comforting someone after a bad presentation.",
    "Words of wisdom for a new parent.", "Advice for overcoming a fear of public speaking.",
    "How to support a friend with burnout?", "Encouraging a student before an exam.",
    "A letter of reconciliation after a fight.", "Comforting a survivor of a natural disaster.",
    "Words for someone struggling with their identity.", "Advice for finding purpose after retirement.",
    "How to celebrate a friend's recovery?", "Encouragement for someone writing their first book.",
    "Supporting someone through a career change.", "Words for someone whose dream was rejected.",
    "How to apologize for a late project?", "Comforting a homesick student.",
    "Encouraging someone to seek mental health help.", "Celebrating a friend's sobriety milestone.",
    "Advice for someone feeling 'stuck' in life.", "How to tell someone you're proud of them.",
    "Comforting a small business owner who failed.", "Words for a couple getting a divorce.",
    "Encouragement for someone learning a difficult skill.", "How to welcome a new neighbor?",
    "Advice for dealing with a 'toxic' friendship.", "Supporting a friend through a health scare.",
    "Words for someone who feels they aren't 'enough'."
]

# COMBINE ALL
all_prompts = []
for i, p in enumerate(factual): all_prompts.append([i+1, "Factual explanation", p])
for i, p in enumerate(opinion): all_prompts.append([i+41, "Opinion & argumentation", p])
for i, p in enumerate(creative): all_prompts.append([i+81, "Creative writing", p])
for i, p in enumerate(code): all_prompts.append([i+121, "Code explanation", p])
for i, p in enumerate(emotional): all_prompts.append([i+161, "Emotional / empathetic response", p])

# SAVE TO CSV
with open('data/raw/prompts.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['prompt_id', 'category', 'text'])
    writer.writerows(all_prompts)

print("SUCCESS: 200 prompts saved to data/raw/prompts.csv")
