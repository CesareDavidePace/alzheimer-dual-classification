import subprocess

def kill_gpu_processes():
    # Run the fuser command to list processes using /dev/nvidia0
    fuser_command = "fuser -v /dev/nvidia1"
    result = subprocess.run(fuser_command.split(), capture_output=True, text=True)

    print(result.stdout)

    # Check if the command was successful
    if result.returncode != 0:
        print(f"Error running fuser command: {result.stderr}")
        return

    # Parse the output to get PIDs
    lines = result.stdout.splitlines()
    pids = []
    for line in lines:
        pids = line.split()

    # Kill each pids
    for pid in pids:
        try:
            kill_command = f"kill -9 {pid}"
            subprocess.run(kill_command.split())
            print(f"Killed process {pid}")
        except Exception as e:
            print(f"Failed to kill process {pid}: {e}")

# Call the function
kill_gpu_processes()