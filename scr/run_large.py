"""Same implementation: sparse incidences and sampled-node InfoNCE are default."""
try:
    from .run import main
except ImportError:
    from run import main

if __name__ == "__main__":
    main()
