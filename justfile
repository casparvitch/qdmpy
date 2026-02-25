# QDMPy - Quantum Diamond MicroscoPy

# Clean build artifacts and caches
clean:
	rm -rf build dist .eggs *.egg-info
	rm -rf .ruff_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -f prospector.log

# Clean everything including virtual environment
clean-all: clean
	rm -rf .venv

# Install in editable mode
install:
	uv pip install -e .

# Install with GUI dependencies
install-gui:
	uv pip install -e .[gui]

# Install with dev dependencies (prospector, etc.)
install-dev:
	uv pip install -e .[dev]

# Install with all extras
install-all:
	uv pip install -e .[gui,dev]

# Install cpufit from local wheel (platform-specific)
install-cpufit:
	uv run python install_gpufit_wheels.py cpufit

# Install gpufit from local wheel (Windows only)
install-gpufit:
	uv run python install_gpufit_wheels.py gpufit

# Install both cpufit and gpufit (where available)
install-fit-backends:
	uv run python install_gpufit_wheels.py both

# Build wheel and sdist
build:
	rm -rf dist
	uv build

# Quick smoke test - can we import it and check version?
check:
	uv run python -c "import qdmpy; print(f'qdmpy {qdmpy.__version__}')"

# Run prospector - always succeeds, check prospector.log for issues
lint: install-dev
	uv run prospector --profile qdmpy.prospector.yaml -o grouped:prospector.log || true

# Run prospector - fails if issues found (for CI)
lint-strict: install-dev
	uv run prospector --profile qdmpy.prospector.yaml -o grouped:prospector.log

# Build documentation
docs:
	uv run pdoc3 --output-dir docs --html --template-dir ./docs/ --force ./src/qdmpy


# Format code with ruff
format:
	uv run ruff format src/qdmpy

# Check code formatting with ruff (no changes)
check-format:
	uv run ruff check --select E,W,F,I src/qdmpy

# Auto-fix issues with ruff
fix:
	uv run ruff check --fix src/qdmpy
