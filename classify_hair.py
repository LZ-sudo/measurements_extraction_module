"""
Combined Hair Segmentation and Classification Pipeline

This script combines hair segmentation and classification:
1. Segments hair from an input image and creates an isolated hair PNG
2. Classifies the isolated hair for both type and color
3. Outputs results to a JSON file

Usage:
    # Basic usage (reads model paths from config.yaml)
    python classify_hair.py path/to/image.jpg --output results.json

    # Save the isolated hair image
    python classify_hair.py path/to/image.jpg --output results.json --save-isolated hair.png

    # Override config with command-line arguments
    python classify_hair.py path/to/image.jpg --output results.json \
        --checkpoint-type hair_classification_model/checkpoints_hair_type/best_model.pth \
        --checkpoint-color hair_classification_model/checkpoints_hair_color/best_model.pth
"""

import argparse
import json
import os
import sys
import tempfile
import torch
import yaml

# Add subdirectories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hair_classification_model'))

from mediapipe_detection.hair_segmentation import segment_hair
from hair_classification_model.src.model import load_model
from hair_classification_model.src.predict import HairPredictor
from hair_classification_model.src.dataset import get_class_names


def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), 'hair_classification_model', 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: config.yaml not found at {config_path}. You must provide all arguments via command line.")
        return None


def load_classification_models(checkpoint_type, checkpoint_color, data_dir_type, data_dir_color, device):
    """Load both hair type and color classification models"""

    print("Loading classification models...")

    # Get class names for both tasks
    type_classes = get_class_names(data_dir_type)
    color_classes = get_class_names(data_dir_color)

    print(f"  Hair Type Classes: {type_classes}")
    print(f"  Hair Color Classes: {color_classes}")

    # Load hair type model
    model_type = load_model(
        checkpoint_type,
        num_classes=len(type_classes),
        model_name='convnext_tiny_in22k'
    )
    predictor_type = HairPredictor(model_type, device, classes=type_classes)

    # Load hair color model
    model_color = load_model(
        checkpoint_color,
        num_classes=len(color_classes),
        model_name='convnext_tiny_in22k'
    )
    predictor_color = HairPredictor(model_color, device, classes=color_classes)

    print("✓ Classification models loaded successfully\n")

    return predictor_type, predictor_color


def classify_hair_from_image(
    image_path,
    output_json_path=None,
    isolated_hair_save_path=None,
    checkpoint_type=None,
    checkpoint_color=None,
    data_dir_type=None,
    data_dir_color=None,
    use_face_detection=True,
    verbose=True
):
    """
    Complete pipeline: segment hair, classify, and save results

    Args:
        image_path: Path to input image
        output_json_path: Path to save JSON results (optional)
        isolated_hair_save_path: Path to save isolated hair PNG (optional)
        checkpoint_type: Path to hair type model checkpoint
        checkpoint_color: Path to hair color model checkpoint
        data_dir_type: Data directory for hair type classes
        data_dir_color: Data directory for hair color classes
        use_face_detection: Whether to use face detection for segmentation
        verbose: Whether to print detailed output

    Returns:
        Dictionary with classification results
    """

    # Step 1: Segment hair and create isolated image
    if verbose:
        print("="*70)
        print("STEP 1: Hair Segmentation")
        print("="*70)
        print(f"Processing image: {image_path}")

    # Create temporary file for isolated hair if not saving permanently
    if isolated_hair_save_path:
        temp_hair_path = isolated_hair_save_path
        cleanup_temp = False
    else:
        temp_fd, temp_hair_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd)
        cleanup_temp = True

    try:
        # Segment and isolate hair
        hair_data = segment_hair(
            image_path,
            use_face_detection=use_face_detection,
            isolated_hair_path=temp_hair_path
        )

        if verbose:
            print(f"✓ Hair segmentation complete")
            if isolated_hair_save_path:
                print(f"✓ Isolated hair saved to: {isolated_hair_save_path}")
            print()

        # Step 2: Load classification models
        if verbose:
            print("="*70)
            print("STEP 2: Hair Classification")
            print("="*70)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if verbose:
            print(f"Using device: {device}\n")

        predictor_type, predictor_color = load_classification_models(
            checkpoint_type,
            checkpoint_color,
            data_dir_type,
            data_dir_color,
            device
        )

        # Step 3: Classify the isolated hair
        if verbose:
            print("Classifying isolated hair...")

        result_type = predictor_type.predict(temp_hair_path, data_dir=data_dir_type)
        result_color = predictor_color.predict(temp_hair_path, data_dir=data_dir_color)

        # Combine results
        results = {
            "input_image": os.path.abspath(image_path),
            "hair_type": {
                "predicted_class": result_type['class'],
                "confidence": float(result_type['confidence']),
                "probabilities": {k: float(v) for k, v in result_type['probabilities'].items()}
            },
            "hair_color": {
                "predicted_class": result_color['class'],
                "confidence": float(result_color['confidence']),
                "probabilities": {k: float(v) for k, v in result_color['probabilities'].items()}
            },
            "combined_confidence": float((result_type['confidence'] + result_color['confidence']) / 2)
        }

        # Add isolated hair path if saved
        if isolated_hair_save_path:
            results["isolated_hair_image"] = os.path.abspath(isolated_hair_save_path)

        # Print results if verbose
        if verbose:
            print(f"\n{'='*70}")
            print("CLASSIFICATION RESULTS")
            print(f"{'='*70}\n")

            print(f"🧵 HAIR TYPE: {results['hair_type']['predicted_class']}")
            print(f"   Confidence: {results['hair_type']['confidence']:.2%}")
            print(f"   Probabilities:")
            for class_name, prob in sorted(results['hair_type']['probabilities'].items(),
                                          key=lambda x: x[1], reverse=True):
                bar = '█' * int(prob * 30)
                print(f"     {class_name:12} {prob:6.2%} {bar}")

            print(f"\n🎨 HAIR COLOR: {results['hair_color']['predicted_class']}")
            print(f"   Confidence: {results['hair_color']['confidence']:.2%}")
            print(f"   Probabilities:")
            for class_name, prob in sorted(results['hair_color']['probabilities'].items(),
                                          key=lambda x: x[1], reverse=True):
                bar = '█' * int(prob * 30)
                print(f"     {class_name:12} {prob:6.2%} {bar}")

            print(f"\n📊 SUMMARY: {results['hair_type']['predicted_class'].upper()} + "
                  f"{results['hair_color']['predicted_class'].upper()}")
            print(f"   Combined Confidence: {results['combined_confidence']:.2%}")
            print(f"{'='*70}\n")

        # Save to JSON if requested
        if output_json_path:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            if verbose:
                print(f"✓ Results saved to: {output_json_path}\n")

        return results

    finally:
        # Clean up temporary file if needed
        if cleanup_temp and os.path.exists(temp_hair_path):
            os.remove(temp_hair_path)


def main():
    # Load config first to get defaults
    config = load_config()

    # Set defaults from config if available
    if config and 'classification' in config:
        default_checkpoint_type = config['classification'].get('checkpoint_type', None)
        default_checkpoint_color = config['classification'].get('checkpoint_color', None)
        default_data_dir_type = config['classification'].get('data_dir_type', './data_hair_type')
        default_data_dir_color = config['classification'].get('data_dir_color', './data_hair_color')

        # Make paths absolute relative to hair_classification_model directory
        base_dir = os.path.join(os.path.dirname(__file__), 'hair_classification_model')
        if default_checkpoint_type and not os.path.isabs(default_checkpoint_type):
            default_checkpoint_type = os.path.join(base_dir, default_checkpoint_type)
        if default_checkpoint_color and not os.path.isabs(default_checkpoint_color):
            default_checkpoint_color = os.path.join(base_dir, default_checkpoint_color)
        if default_data_dir_type and not os.path.isabs(default_data_dir_type):
            default_data_dir_type = os.path.join(base_dir, default_data_dir_type)
        if default_data_dir_color and not os.path.isabs(default_data_dir_color):
            default_data_dir_color = os.path.join(base_dir, default_data_dir_color)
    else:
        default_checkpoint_type = None
        default_checkpoint_color = None
        default_data_dir_type = None
        default_data_dir_color = None

    parser = argparse.ArgumentParser(
        description='Combined Hair Segmentation and Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (reads from config.yaml)
  python classify_hair.py test.jpg --output results.json

  # Save isolated hair image
  python classify_hair.py test.jpg --output results.json --save-isolated hair.png

  # Override config values
  python classify_hair.py test.jpg --output results.json \\
      --checkpoint-type hair_classification_model/checkpoints_hair_type/best_model.pth \\
      --checkpoint-color hair_classification_model/checkpoints_hair_color/best_model.pth

Configuration:
  Model paths are read from hair_classification_model/config.yaml by default.
        """
    )

    # Positional argument
    parser.add_argument('image', type=str,
                       help='Path to input image')

    # Optional arguments
    parser.add_argument('-o', '--output', type=str,
                       help='Path to save JSON results (optional)')

    # Optional arguments
    parser.add_argument('--save-isolated', type=str,
                       help='Path to save isolated hair PNG (optional)')
    parser.add_argument('--no-face-detection', action='store_true',
                       help='Disable face detection for segmentation (use for close-up head shots)')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output (only errors)')

    # Model configuration (optional if set in config)
    parser.add_argument('--checkpoint-type', type=str, default=default_checkpoint_type,
                       help='Path to hair type model checkpoint')
    parser.add_argument('--checkpoint-color', type=str, default=default_checkpoint_color,
                       help='Path to hair color model checkpoint')
    parser.add_argument('--data-dir-type', type=str, default=default_data_dir_type,
                       help='Data directory for hair type classes')
    parser.add_argument('--data-dir-color', type=str, default=default_data_dir_color,
                       help='Data directory for hair color classes')

    args = parser.parse_args()

    # Validate input image exists
    if not os.path.exists(args.image):
        print(f"Error: Input image not found: {args.image}")
        sys.exit(1)

    # Check that checkpoints are specified
    if not args.checkpoint_type:
        print("Error: Hair type checkpoint not specified. Either:")
        print("  1. Set 'classification.checkpoint_type' in hair_classification_model/config.yaml, or")
        print("  2. Provide --checkpoint-type argument")
        sys.exit(1)

    if not args.checkpoint_color:
        print("Error: Hair color checkpoint not specified. Either:")
        print("  1. Set 'classification.checkpoint_color' in hair_classification_model/config.yaml, or")
        print("  2. Provide --checkpoint-color argument")
        sys.exit(1)

    # Check that checkpoints exist
    if not os.path.exists(args.checkpoint_type):
        print(f"Error: Hair type checkpoint not found: {args.checkpoint_type}")
        sys.exit(1)

    if not os.path.exists(args.checkpoint_color):
        print(f"Error: Hair color checkpoint not found: {args.checkpoint_color}")
        sys.exit(1)

    # Check that data directories are specified and exist
    if not args.data_dir_type:
        print("Error: Hair type data directory not specified")
        sys.exit(1)

    if not args.data_dir_color:
        print("Error: Hair color data directory not specified")
        sys.exit(1)

    if not os.path.exists(args.data_dir_type):
        print(f"Error: Hair type data directory not found: {args.data_dir_type}")
        sys.exit(1)

    if not os.path.exists(args.data_dir_color):
        print(f"Error: Hair color data directory not found: {args.data_dir_color}")
        sys.exit(1)

    # Run the pipeline
    try:
        results = classify_hair_from_image(
            image_path=args.image,
            output_json_path=args.output,
            isolated_hair_save_path=args.save_isolated,
            checkpoint_type=args.checkpoint_type,
            checkpoint_color=args.checkpoint_color,
            data_dir_type=args.data_dir_type,
            data_dir_color=args.data_dir_color,
            use_face_detection=not args.no_face_detection,
            verbose=not args.quiet
        )

        if args.quiet:
            # Just print the summary in quiet mode
            print(f"{results['hair_type']['predicted_class']} + {results['hair_color']['predicted_class']}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
