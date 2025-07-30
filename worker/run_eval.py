import argparse
from typing import Optional
import datasets
import evaluate
import tempfile
import time
import os
import requests
import itertools
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
import google.genai.types as genai_types
import base64
from pathlib import Path
import litellm
import csv
import pandas as pd

load_dotenv()


def fetch_audio_urls(dataset_path, dataset, split, batch_size=100, max_retries=20):
    API_URL = "https://datasets-server.huggingface.co/rows"

    size_url = f"https://datasets-server.huggingface.co/size?dataset={dataset_path}&config={dataset}&split={split}"
    size_response = requests.get(size_url).json()
    total_rows = size_response["size"]["config"]["num_rows"]
    audio_urls = []
    for offset in tqdm(range(0, total_rows, batch_size), desc="Fetching audio URLs"):
        params = {
            "dataset": dataset_path,
            "config": dataset,
            "split": split,
            "offset": offset,
            "length": min(batch_size, total_rows - offset),
        }

        retries = 0
        while retries <= max_retries:
            try:
                headers = {}
                if os.environ.get("HF_TOKEN") is not None:
                    headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
                else:
                    print("HF_TOKEN not set, might experience rate-limiting.")
                response = requests.get(API_URL, params=params)
                response.raise_for_status()
                data = response.json()
                yield from data["rows"]
                break
            except (requests.exceptions.RequestException, ValueError) as e:
                retries += 1
                print(
                    f"Error fetching data: {e}, retrying ({retries}/{max_retries})..."
                )
                time.sleep(10)
                if retries >= max_retries:
                    raise Exception("Max retries exceeded while fetching data.")


def transcribe_with_retry(
    model_name: str,
    audio_file_path: Optional[str],
    sample: dict,
    max_retries=10,
    use_url=False,
):
    retries = 0
    while retries <= max_retries:
        try:
            if model_name.startswith("google/"):
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                # The genai.Client picks API key from env var GEMINI_API_KEY by default, but allow override
                client = genai.Client(api_key=api_key) if api_key else genai.Client()

                model_name_only = model_name.split("/")[1]

                # Prepare audio bytes and MIME type
                if use_url:
                    audio_url = sample["row"]["audio"][0]["src"]
                    response = requests.get(audio_url)
                    response.raise_for_status()
                    audio_bytes = response.content

                    mime_type = response.headers.get("Content-Type")
                    if not mime_type:
                        import mimetypes
                        mime_type, _ = mimetypes.guess_type(audio_url)
                        if not mime_type:
                            mime_type = "audio/wav"  # sensible default
                else:
                    import mimetypes
                    with open(audio_file_path, "rb") as f:
                        audio_bytes = f.read()
                    mime_type, _ = mimetypes.guess_type(audio_file_path)
                    if not mime_type:
                        mime_type = "audio/wav"  # default

                # Create Part object from audio bytes. Newer versions expose Part.from_bytes,
                # but older versions might require manual construction via Blob/Part.
                if hasattr(genai_types, "Part") and hasattr(genai_types.Part, "from_bytes"):
                    audio_part = genai_types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
                else:
                    # Fallback manual construction
                    if not hasattr(genai_types, "Blob"):
                        raise RuntimeError("google-generativeai package is outdated. Please upgrade to >=0.5.0")
                    audio_part = genai_types.Part(
                        inline_data=genai_types.Blob(data=audio_bytes, mime_type=mime_type)
                    )

                prompt = "Generate a transcript of the speech."
                response = client.models.generate_content(
                    model=model_name_only,
                    contents=[prompt, audio_part],
                )
                return response.text.strip()

            elif model_name.startswith("litellm/"):
                import mimetypes
                model_name_only = model_name[len("litellm/"):]

                # Prepare audio bytes and MIME type
                if use_url:
                    audio_url = sample["row"]["audio"][0]["src"]
                    response = requests.get(audio_url)
                    response.raise_for_status()
                    audio_bytes = response.content
                    mime_type = response.headers.get("Content-Type")
                    if not mime_type:
                        mime_type, _ = mimetypes.guess_type(audio_url)
                        if not mime_type:
                            mime_type = "audio/wav"  # default fallback
                else:
                    with open(audio_file_path, "rb") as f:
                        audio_bytes = f.read()
                    mime_type, _ = mimetypes.guess_type(audio_file_path)
                    if not mime_type:
                        mime_type = "audio/wav"

                # Encode audio to base64 for LiteLLM
                encoded_data = base64.b64encode(audio_bytes).decode("utf-8")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Generate a transcript of the speech."},
                            {"type": "file", "file": {"file_data": f"data:{mime_type};base64,{encoded_data}"}},
                        ],
                    }
                ]

                # Build model string with proxy prefix to satisfy LiteLLM provider detection
                proxy_model = f"litellm_proxy/{model_name_only}"
                response = litellm.completion(
                    model=proxy_model,
                    messages=messages,
                    api_base="http://litellm:4000",
                    api_key="dummy",  # proxy ignores, but litellm SDK expects a value
                )

                # LiteLLM returns OpenAI-style response objects; extract content robustly
                try:
                    transcript_text = response["choices"][0]["message"]["content"]
                except Exception:
                    transcript_text = str(response)
                return transcript_text.strip()

            else:
                raise ValueError(
                    "Invalid model prefix, must start with 'google/' or 'litellm/''"
                )

        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise e

            if not use_url:
                sf.write(
                    audio_file_path,
                    sample["audio"]["array"],
                    sample["audio"]["sampling_rate"],
                    format="WAV",
                )
            delay = 1
            print(
                f"API Error: {str(e)}. Retrying in {delay}s... (Attempt {retries}/{max_retries})"
            )
            time.sleep(delay)


def transcribe_dataset(
    dataset_path,
    dataset,
    split,
    model_name,
    use_url=False,
    max_samples=None,
    max_workers=4,
):
    if use_url:
        audio_rows = fetch_audio_urls(dataset_path, dataset, split)
        if max_samples:
            audio_rows = itertools.islice(audio_rows, max_samples)
        ds = audio_rows
    else:
        ds = datasets.load_dataset(dataset_path, dataset, split=split, streaming=False)
        ds = data_utils.prepare_data(ds)
        if max_samples:
            ds = ds.take(max_samples)

    results = {
        "references": [],
        "predictions": [],
        "audio_length_s": [],
        "transcription_time_s": [],
    }

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    model_name_safe = model_name.replace("/", "_")
    output_path = output_dir / f"{dataset}_{split}_{model_name_safe}.csv"

    # Open file and write header
    with open(output_path, 'w', newline='', buffering=1) as csvfile:
        fieldnames = ["dataset", "split", "reference", "transcript", "wer", "rfx"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, sample in enumerate(ds):
            start_time = time.time()
            reference = ""
            try:
                if use_url:
                    transcript = transcribe_with_retry(
                        model_name, None, sample, use_url=True
                    )
                    reference = sample["row"]["text"].strip() or " "
                    audio_duration = sample["row"]["audio_length_s"]
                else:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
                        sf.write(
                            tmpfile.name,
                            sample["audio"]["array"],
                            sample["audio"]["sampling_rate"],
                        )
                        transcript = transcribe_with_retry(model_name, tmpfile.name, sample)
                    reference = sample.get("norm_text", "").strip() or " "
                    audio_duration = (len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"])

                end_time = time.time()
                processing_time = end_time - start_time
                rfx = processing_time / audio_duration if audio_duration > 0 else 0

                normalized_reference = data_utils.normalizer(reference)
                normalized_transcript = data_utils.normalizer(transcript)

                wer_metric = evaluate.load("wer")
                if not normalized_reference.strip():
                    # If reference is empty, WER is 1.0 if transcript is not empty, 0.0 if both are empty.
                    wer = 1.0 if normalized_transcript.strip() else 0.0
                else:
                    wer = wer_metric.compute(
                        predictions=[normalized_transcript],
                        references=[normalized_reference],
                    )

                result_row = {
                    "dataset": dataset,
                    "split": split,
                    "reference": normalized_reference,
                    "transcript": normalized_transcript,
                    "wer": wer,
                    "rfx": rfx,
                }
                writer.writerow(result_row)
                print(f"Processed sample {i+1}, WER: {wer:.4f}, RFx: {rfx:.4f}")

            except Exception as e:
                print(f"Error processing sample {i+1}: {e}")
                # Optionally, log the error to the CSV
                error_row = {
                    "dataset": dataset,
                    "split": split,
                    "reference": data_utils.normalizer(reference) if reference else 'UNKNOWN',
                    "transcript": f"ERROR: {e}",
                    "wer": 1.0,  # Assign max WER for errors
                    "rfx": float("inf"),
                }
                writer.writerow(error_row)

    # After loop, load all results from the CSV for final calculation
    results_df = pd.read_csv(output_path)
    # Filter out any rows that were errors
    results_df = results_df[results_df["wer"] != float("inf")]

    final_references = results_df["reference"].astype(str).tolist()
    final_predictions = results_df["transcript"].astype(str).tolist()

    manifest_path = data_utils.write_manifest(
        final_references,
        final_predictions,
        model_name.replace("/", "-"),
        dataset_path,
        dataset,
        split,
    )
    print(f"Results manifest saved at path: {manifest_path}")

    # Final WER calculation with protection against empty references
    wer_metric = evaluate.load("wer")
    valid_pairs = [(p, r) for p, r in zip(final_predictions, final_references) if str(r).strip()]
    
    if not valid_pairs:
        wer = 1.0 if any(p.strip() for p in final_predictions) else 0.0
        print("No valid reference transcripts found to calculate final WER.")
    else:
        valid_predictions, valid_references = zip(*valid_pairs)
        wer = wer_metric.compute(predictions=list(valid_predictions), references=list(valid_references))
    
    # Calculate average RFx from the successful runs
    if not results_df.empty:
        rfx = results_df["rfx"].mean()
    else:
        rfx = 0.0

    print(f"Final WER: {wer:.4f}, Average RFx: {rfx:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Transcription Script with Concurrency"
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--model_name",
        required=True,
        help="Prefix model name with 'assembly/', 'openai/', 'elevenlabs/', 'revai/', 'speechmatics/', or 'google/'",
    )
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument(
        "--max_workers", type=int, default=300, help="Number of concurrent threads"
    )
    parser.add_argument(
        "--use_url",
        action="store_true",
        help="Use URL-based audio fetching instead of datasets",
    )

    args = parser.parse_args()

    # If --max_samples not provided, look for env variable
    if args.max_samples is None:
        env_max = os.getenv("MAX_SAMPLES")
        if env_max and env_max.strip():
            try:
                args.max_samples = int(env_max)
                print(f"Using MAX_SAMPLES from environment: {args.max_samples}")
            except ValueError:
                print(f"Ignoring invalid MAX_SAMPLES env var value: {env_max}")

    transcribe_dataset(
        dataset_path=args.dataset_path,
        dataset=args.dataset,
        split=args.split,
        model_name=args.model_name,
        use_url=args.use_url,
        max_samples=args.max_samples,
        max_workers=args.max_workers,
    )
