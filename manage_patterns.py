import json
from patterns.pattern_manager import PatternManager

def display_menu():
    print("Pattern Management")
    print("1. Add Pattern")
    print("2. View All Patterns")
    print("3. View All Patterns (JSON)")
    print("4. View HW Patterns (JSON)")
    print("5. View SW Patterns (JSON)")
    print("6. Remove Pattern")
    print("7. Exit")

def add_pattern(pattern_manager):
    label = input("Enter label (HW/SW): ").strip().upper()
    if label not in ['HW', 'SW']:
        print("Invalid label. Please enter 'HW' or 'SW'.")
        return
    pattern = input("Enter pattern: ").strip()
    pattern_manager.add_pattern(label, pattern)
    pattern_manager.learn_from_text(pattern, label)  # Learn from the new pattern
    print(f"Pattern added to {label} category.")

def view_patterns(pattern_manager):
    print("Current Patterns:")
    for label, patterns in pattern_manager.patterns.items():
        print(f"{label} Patterns:")
        for pattern in patterns:
            print(f"  - {pattern}")

def view_patterns_json(pattern_manager):
    print("Current Patterns in JSON format:")
    print(json.dumps(pattern_manager.patterns, indent=4))

def view_hw_patterns_json(pattern_manager):
    print("HW Patterns in JSON format:")
    print(json.dumps({"HW": pattern_manager.patterns["HW"]}, indent=4))

def view_sw_patterns_json(pattern_manager):
    print("SW Patterns in JSON format:")
    print(json.dumps({"SW": pattern_manager.patterns["SW"]}, indent=4))



def main():
    pattern_manager = PatternManager()

    while True:
        display_menu()
        choice = input("Enter your choice: ").strip()
        if choice == '1':
            add_pattern(pattern_manager)
        elif choice == '2':
            view_patterns(pattern_manager)
        elif choice == '3':
            view_patterns_json(pattern_manager)
        elif choice == '4':
            view_hw_patterns_json(pattern_manager)
        elif choice == '5':
            view_sw_patterns_json(pattern_manager)
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    main()
