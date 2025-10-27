import vertexai
import json
import time
from pathlib import Path
from vertexai.generative_models import GenerativeModel, Part, Image

# CONFIGURATION 

PROJECT_ID = "Enter your personal project ID"

#  A valid Google Cloud location
LOCATION = "us-central1"


ROOT_BUNDLES_DIR = Path('/home/afaris/Final_Project/gemini_bundles')

#  Define the model to use
MODEL_NAME = "gemini-2.5-pro"


#  SCRIPT START 

def process_bundle(bundle_path: Path, project_id: str, location: str):
    vertexai.init(project=project_id, location=location)

    print(f"  -> Loading data from: {bundle_path.name}")
    try:
        with open(bundle_path / 'commands.json', 'r') as f:
            commands = json.load(f)
        images = [Image.load_from_file(bundle_path / f"frame_{i}.jpg") for i in range(6)]
    except FileNotFoundError:
        print(f"  ❌ ERROR: Could not find 'commands.json' or all 6 frame images in '{bundle_path}'.")
        return False

    # The prompt for the model
    prompt = """
### ROLE ###
You are an expert robotics analyst specializing in social navigation and human-robot interaction.

### TASK ###
Analyze the provided sequence of 6 keyframes and their corresponding velocity commands from a ground robot's perspective. Your analysis should be structured as a sequence of 5 transitions (Frame 1-to-2, 2-to-3, etc.). For each transition, describe the robot's action, justify why it was a good action in that context, and list which social navigation principles were respected, explaining your reasoning.

### GUIDING PRINCIPLE ###
Your primary directive is to be a factual observer. Describe **only what is explicitly visible** in the frames. Prioritize literal description over narrative inference. Do not predict outcomes or describe events that have not yet visibly occurred.

### CONTROL CONVENTION ###
The velocity commands follow these rules:
- **Positive `linear_x`**:  Indicates forward motion.
- **Negative `linear_x`**: Indicates backward motion.
- **Positive `angular_z`**: Indicates a counter-clockwise turn (to the robot's left).
- **Negative `angular_z`**: Indicates a clockwise turn (to the robot's right).

### CONTEXT: 8 PRINCIPLES OF SOCIAL NAVIGATION ###
These are the principles to consider for your analysis:
1.  **P1: Safety:** Avoid damage to humans, robots, or the environment.
2.  **P2: Comfort:** Avoid causing annoyance or stress in humans.
3.  **P3: Legibility:** Behave so the robot's goals can be easily understood by others.
4.  **P4: Politeness:** Be respectful and considerate of other agents.
5.  **P5: Social Norms:** Comply with common social rules for navigating in shared space.
6.  **P6: Agent Understanding:** Predict and accommodate the behavior of other agents.
7.  **P7: Proactivity:** Take the initiative to resolve potential deadlocks or ambiguities.
8.  **P8: Contextual Appropriateness:** Behave properly in the current context.

### DATA SEQUENCE ###
Here is the sequence of 6 keyframes and the robot's velocity command at each moment.
"""
    
    prompt_parts = [prompt]
    for i in range(6):
        frame_name = f"frame_{i}.jpg"
        command_data = commands.get(frame_name, {})
        prompt_parts.append(f"\n**Frame {i+1}:**")
        prompt_parts.append(images[i])
        prompt_parts.append(f"Command {i+1}: [{command_data.get('linear_x', 0):.4f}, {command_data.get('angular_z', 0):.4f}]")

    final_instructions = """
### INSTRUCTIONS ###
Provide your complete analysis as a JSON object which is a list of 5 analyses, one for each frame-to-frame transition. Follow the format below exactly. Do not add any text before or after the JSON object.

[
  {
    "sequence": "Frame 1 to Frame 2",
    "scene_description": "Describe the scene and how it changes between frame 1 and frame 2.",
    "high_level_command": "A high-level description of the robot's action (e.g., 'The robot accelerates forward while turning right.').",
    "action_justification": "Explain why this action was the best one in this context, based on the visual evidence.",
    "respected_principles": [
      {
        "principle": "Name of the respected principle (e.g., P1: Safety)",
        "justification": "A concise explanation of how the robot's action in this sequence respected this principle."
      }
    ]
  }
]
"""
    prompt_parts.append(final_instructions)

    print(f"  -> Sending request to '{MODEL_NAME}'...")
    try:
        model = GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt_parts)
        
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        output_data = json.loads(cleaned_response)
        
        json_output_path = bundle_path / 'analysis.json'
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)

        print(f"  ✅ Analysis saved to: {json_output_path}")
        return True

    except Exception as e:
        print(f"  ❌ ERROR on bundle {bundle_path.name}: {e}")
        return False

# AUTOMATION SCRIPT MAIN EXECUTION 
if __name__ == "__main__":
    print(f"--- Starting Gemini Automation Pipeline ---")
    print(f"Searching for bundles in: {ROOT_BUNDLES_DIR}\n")
    
    # 1. Find all bundle folders (b1, b2, etc.)
    if not ROOT_BUNDLES_DIR.is_dir():
        print(f"❌ ERROR: Root bundles directory not found at '{ROOT_BUNDLES_DIR}'")
        exit()
        
    bundle_paths = [p for p in ROOT_BUNDLES_DIR.iterdir() if p.is_dir() and p.name.startswith('b')]
    
    # 2. Sort the bundles numerically to process them in order
    sorted_bundle_paths = sorted(bundle_paths, key=lambda p: int(p.name[1:]))
    
    total_bundles = len(sorted_bundle_paths)
    if total_bundles == 0:
        print("No bundles found to process.")
        exit()
        
    print(f"Found {total_bundles} bundles to process.\n")
    
    success_count = 0
    fail_count = 0
    
    # 3. Loop through each bundle and process it
    for i, path in enumerate(sorted_bundle_paths):
        print(f"--- Processing Bundle {i+1} of {total_bundles}: {path.name} ---")
        
        # Check if analysis already exists to allow for resuming
        if (path / 'analysis.json').is_file():
            print("  -> Analysis file already exists. Skipping.")
            success_count += 1
            continue

        if process_bundle(path, PROJECT_ID, LOCATION):
            success_count += 1
        else:
            fail_count += 1
        
        # Add a delay between API calls to respect rate limits
        print("  -> Waiting for 2 seconds before next request...")
        time.sleep(2)

    print(f"\n{'='*20} PIPELINE COMPLETE {'='*20}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process: {fail_count}")
    print(f"Total bundles: {total_bundles}")