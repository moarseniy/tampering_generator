#!/usr/bin/env python3
import argparse
from core import DocumentForgeryGenerator

def main():
    parser = argparse.ArgumentParser(description='Document Forgery Generator')
    parser.add_argument('--config', type=str, default='configs/generator_config.yaml',
                       help='Path to configuration file')

    args = parser.parse_args()
    
    # Инициализация генератора
    generator = DocumentForgeryGenerator(args.config)
    
    # Запуск генерации
    generator.generate_dataset()

if __name__ == "__main__":
    main()
