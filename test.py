# Coverage Requirements per Day, Shift, and Procedure
coverage_requirements = {
    "Sunday": {
        "p1": {"Morning": 4, "Afternoon": 3, "Night": 2},
        "p2": {"Morning": 5, "Afternoon": 2, "Night": 1},
        "p3": {"Morning": 3, "Afternoon": 3, "Night": 0},
        "p4": {"Morning": 3, "Afternoon": 3, "Night": 0},
        "p5": {"Morning": 4, "Afternoon": 4, "Night": 0},
    },
    "Monday": {
        "p1": {"Morning": 5, "Afternoon": 4, "Night": 3},
        "p2": {"Morning": 5, "Afternoon": 4, "Night": 2},
        "p3": {"Morning": 5, "Afternoon": 4, "Night": 1},
        "p4": {"Morning": 5, "Afternoon": 4, "Night": 1},
        "p5": {"Morning": 6, "Afternoon": 6, "Night": 1},
    },
    "Tuesday": {
        "p1": {"Morning": 5, "Afternoon": 4, "Night": 3},
        "p2": {"Morning": 5, "Afternoon": 4, "Night": 2},
        "p3": {"Morning": 5, "Afternoon": 4, "Night": 1},
        "p4": {"Morning": 5, "Afternoon": 4, "Night": 1},
        "p5": {"Morning": 6, "Afternoon": 6, "Night": 1},
    },
    "Wednesday": {
        "p1": {"Morning": 5, "Afternoon": 4, "Night": 3},
        "p2": {"Morning": 5, "Afternoon": 4, "Night": 2},
        "p3": {"Morning": 5, "Afternoon": 4, "Night": 1},
        "p4": {"Morning": 5, "Afternoon": 4, "Night": 1},
        "p5": {"Morning": 6, "Afternoon": 6, "Night": 1},
    },
    "Thursday": {
        "p1": {"Morning": 5, "Afternoon": 4, "Night": 3},
        "p2": {"Morning": 5, "Afternoon": 4, "Night": 2},
        "p3": {"Morning": 5, "Afternoon": 4, "Night": 1},
        "p4": {"Morning": 5, "Afternoon": 4, "Night": 1},
        "p5": {"Morning": 6, "Afternoon": 6, "Night": 1},
    },
    "Friday": {
        "p1": {"Morning": 5, "Afternoon": 4, "Night": 3},
        "p2": {"Morning": 5, "Afternoon": 4, "Night": 2},
        "p3": {"Morning": 5, "Afternoon": 4, "Night": 1},
        "p4": {"Morning": 5, "Afternoon": 4, "Night": 1},
        "p5": {"Morning": 6, "Afternoon": 6, "Night": 1},
    },
    "Saturday": {
        "p1": {"Morning": 4, "Afternoon": 3, "Night": 2},
        "p2": {"Morning": 5, "Afternoon": 2, "Night": 1},
        "p3": {"Morning": 3, "Afternoon": 3, "Night": 0},
        "p4": {"Morning": 3, "Afternoon": 3, "Night": 0},
        "p5": {"Morning": 4, "Afternoon": 4, "Night": 0},
    },
}

# Calculate the sum of people needed per day per shift
shift_totals = {}

for day, procedures in coverage_requirements.items():
    shift_totals[day] = {"Morning": 0, "Afternoon": 0, "Night": 0}
    for procedure, shifts in procedures.items():
        for shift, count in shifts.items():
            shift_totals[day][shift] += count

# Print results
for day, shifts in shift_totals.items():
    print(
        f"{day}: Morning: {shifts['Morning']}, Afternoon: {shifts['Afternoon']}, Night: {shifts['Night']}"
    )
