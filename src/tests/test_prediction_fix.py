import os
import sys
import asyncio

# Add project root to path Soo that no erro would come that src not found
sys.path.append(os.getcwd())

from src.pipelines.Prediction_Pipeline import PredictionPipeline

async def main():
    try:
        prediction_pipeline = PredictionPipeline()
        result = await prediction_pipeline.initiate_prediction_pipeline("give application accessibility workout")
        print(result)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    asyncio.run(main())