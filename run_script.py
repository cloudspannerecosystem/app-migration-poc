import sys
import asyncio
from app_migrator_analysis import MigrationSummarizer

async def generate_migration_report(source_directory, access_key, output_file, gemini_version):
    # Example implementation
    summarizer = MigrationSummarizer(access_key, gemini_version)

    summaries = await summarizer.analyze_project(source_directory)
    
    await summarizer.summarize_report(summaries, output_file)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python path_to_your_python_script.py <source_directory> <access_key> <output_file> [<gemini_version>]")
        sys.exit(1)
    
    source_directory = sys.argv[1]
    access_key = sys.argv[2]
    output_file = sys.argv[3]
    gemini_version = sys.argv[4] if len(sys.argv) > 4 else "gemini-1.5-flash-001"

    asyncio.run(generate_migration_report(source_directory, access_key, output_file, gemini_version))




