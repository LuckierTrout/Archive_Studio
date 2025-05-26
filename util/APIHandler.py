# util/APIHandler.py

# This file contains the APIHandler class, which is used to handle
# the API calls for the application.

import asyncio
import base64
import os
from pathlib import Path
from PIL import Image

# OpenAI API
from openai import OpenAI
import openai

# Anthropic API
from anthropic import AsyncAnthropic
import anthropic

# Google API
import google.generativeai as genai_client
from google.generativeai import types

# Import ErrorLogger
from util.ErrorLogger import log_error

class APIHandler:
    def __init__(self, openai_api_key, anthropic_api_key, google_api_key, app=None):
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.google_api_key = google_api_key
        self.app = app  # Reference to main app for error logging
        
    def log_error(self, error_message, additional_info=None):
        """Log errors using ErrorLogger if app is available, otherwise silently continue"""
        if self.app and hasattr(self.app, 'base_dir') and hasattr(self.app, 'log_level'):
            log_error(self.app.base_dir, self.app.log_level, error_message, additional_info, level="ERROR")
        
    async def route_api_call(self, engine, system_prompt, user_prompt, temp, 
                           image_data=None, text_to_process=None, val_text=None, 
                           index=None, is_base64=True, formatting_function=False, 
                           api_timeout=80, job_type=None, job_params=None):
        """
        Routes the API call to the appropriate service based on the engine name.
        
        Args:
            engine: Model to use (gpt/claude/gemini)
            system_prompt: System instructions for the AI
            user_prompt: User instructions for the AI
            temp: Temperature setting for model output
            image_data: Optional image data (single image or list of tuples)
            text_to_process: Text to be processed and inserted into prompt
            val_text: Validation text to check in response
            index: Document index for tracking
            is_base64: Whether images are base64 encoded
            formatting_function: Whether to use user_prompt directly or format it
            api_timeout: Timeout in seconds for API call
            job_type: Type of job (e.g., "Metadata")
            job_params: Additional parameters for the job
        """
        # Extract required headers for metadata validation if applicable
        required_headers = job_params.get("required_headers") if job_type == "Metadata" and job_params else None
        
        # Debug print for image context
        if image_data:
            if isinstance(image_data, list):
                try:
                    print(f"[DEBUG] APIHandler.route_api_call received {len(image_data)} images: {[l for _, l in image_data]}")
                except Exception:
                    print(f"[DEBUG] APIHandler.route_api_call received image_data (list) but could not extract labels.")
            else:
                print(f"[DEBUG] APIHandler.route_api_call received image_data of type {type(image_data)}")
        else:
            print(f"[DEBUG] APIHandler.route_api_call received no image_data.")
        
        if "gpt" in engine.lower() or "o1" in engine.lower() or "o3" in engine.lower():
            return await self.handle_gpt_call(system_prompt, user_prompt, temp, 
                                           image_data, text_to_process, val_text, 
                                           engine, index, is_base64, formatting_function, 
                                           api_timeout, job_type, required_headers)
        elif "gemini" in engine.lower():
            return await self.handle_gemini_call(system_prompt, user_prompt, temp, 
                                              image_data, text_to_process, val_text, 
                                              engine, index, is_base64, formatting_function, 
                                              api_timeout, job_type, required_headers)
        elif "claude" in engine.lower():
            return await self.handle_claude_call(system_prompt, user_prompt, temp, 
                                              image_data, text_to_process, val_text, 
                                              engine, index, is_base64, formatting_function, 
                                              api_timeout, job_type, required_headers)
        else:
            raise ValueError(f"Unsupported engine: {engine}")
    
    def _prepare_gpt_messages(self, system_prompt, user_prompt, image_data):
        """Prepare messages for GPT API with system prompt and image content if present"""
        is_o_series_model = "o1" in system_prompt.lower() or "o3" in system_prompt.lower()
        role_key = "developer" if is_o_series_model else "system"
        
        # Base messages without images
        if not image_data:
            return [
                {"role": role_key, "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
            ]
        
        # Single image case
        if isinstance(image_data, str):
            return [
                {"role": role_key, "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "high"
                            },
                        },
                    ],
                }
            ]
        
        # Multiple images case
        content = [{"type": "text", "text": user_prompt}]
        for img, label in image_data:
            if label:
                content.append({"type": "text", "text": label})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img}",
                    "detail": "high"
                }
            })
        
        return [
            {"role": role_key, "content": system_prompt},
            {"role": "user", "content": content}
        ]
    
    async def handle_gpt_call(self, system_prompt, user_prompt, temp, image_data, 
                            text_to_process, val_text, engine, index, 
                            is_base64=True, formatting_function=False, api_timeout=25.0,
                            job_type=None, required_headers=None):
        """Handle API calls to OpenAI GPT models"""
        client = OpenAI(api_key=self.openai_api_key, timeout=api_timeout)
        
        populated_user_prompt = user_prompt if formatting_function else user_prompt.format(text_to_process=text_to_process)
        max_tokens = 2000 if job_type == "Metadata" else (200 if "pagination" in user_prompt.lower() else 1500)
        max_retries = 5 if job_type == "Metadata" else 3
        retries = 0
        is_o_series_model = "o1" in engine.lower() or "o3" in engine.lower()
        
        while retries < max_retries:
            try:
                messages = self._prepare_gpt_messages(system_prompt, populated_user_prompt, image_data)
                
                api_params = {
                    "model": engine,
                    "messages": messages,
                }
                
                # Add model-specific parameters
                if is_o_series_model:
                    api_params["response_format"] = {"type": "text"}
                    api_params["reasoning_effort"] = "low"
                else:
                    api_params["temperature"] = temp
                    api_params["max_tokens"] = max_tokens
                
                message = client.chat.completions.create(**api_params)
                response = message.choices[0].message.content
                validation_result = self._validate_response(response, val_text, index, job_type, required_headers)
                
                # If validation failed, adjust parameters and retry
                if validation_result[0] == "Error" and retries < max_retries - 1:
                    if job_type == "Metadata" and not is_o_series_model:
                        api_params["temperature"] = min(0.9, float(temp) + (retries * 0.1))
                        if retries >= 2:
                            api_params["max_tokens"] = min(4000, max_tokens + 500)
                    
                    retries += 1
                    await asyncio.sleep(1 * (1.5 ** retries))  # Exponential backoff
                    continue
                
                return validation_result

            except (openai.APITimeoutError, openai.APIError) as e:
                self.log_error(f"GPT API Error with {engine} for index {index}", f"{str(e)}")
                retries += 1
                if retries == max_retries:
                    return "Error", index
                await asyncio.sleep(1 * (1.5 ** retries))
    
    async def handle_gemini_call(self, system_prompt, user_prompt, temp, image_data, 
                                text_to_process, val_text, engine, index, 
                                is_base64=True, formatting_function=False, api_timeout=120.0,
                                job_type=None, required_headers=None):
        """Handle API calls to Google Gemini models"""
        
        # 1. Configure the API key 
        genai_client.configure(api_key=self.google_api_key) 
        
        populated_user_prompt = user_prompt if formatting_function else user_prompt.format(text_to_process=text_to_process)
        
        # 2. Define GenerationConfig using google.generativeai.types
        generation_config = types.GenerationConfig( # Ensure 'types' is imported as 'from google.generativeai import types'
            temperature=temp,
            top_p=0.95, 
            top_k=40,   
            max_output_tokens=8192, 
        )

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        max_retries = 5 if job_type == "Metadata" else 3
        retries = 0
        
        current_generation_config = generation_config # Use a mutable copy for retries

        while retries < max_retries:
            try:
                parts = [] 
                
                # Image data handling 
                if image_data:
                    if isinstance(image_data, (str, Path)): 
                        try:
                            # Ensure Path objects are converted to string for Image.open
                            img_path_str = str(image_data) if isinstance(image_data, Path) else image_data
                            img = Image.open(img_path_str)
                            parts.append(img) 
                        except Exception as e:
                            self.log_error(f"Failed to open image for Gemini: {image_data}", str(e))
                    elif isinstance(image_data, list): 
                        for img_item, label in image_data:
                            if label:
                                parts.append(label) 
                            try:
                                if isinstance(img_item, str) and os.path.exists(img_item):
                                    img = Image.open(img_item)
                                    parts.append(img)
                                elif isinstance(img_item, Image.Image): 
                                    parts.append(img_item)
                                # Add handling for base64 encoded images if necessary for Gemini
                                elif isinstance(img_item, str) and is_base64: # Assuming img_item could be base64 string
                                    try:
                                        img_bytes = base64.b64decode(img_item)
                                        img = Image.open(io.BytesIO(img_bytes)) # type: ignore
                                        parts.append(img)
                                    except Exception as e_b64:
                                        self.log_error(f"Failed to decode base64 image for Gemini: {label or 'untitled'}", str(e_b64))
                            except Exception as e:
                                self.log_error(f"Failed to process image for Gemini: {img_item}", str(e))
                
                parts.append(populated_user_prompt) 

                # 3. Initialize the model using GenerativeModel
                model = genai_client.GenerativeModel(
                    model_name=engine,
                    generation_config=current_generation_config, # Use the potentially modified config for retries
                    safety_settings=safety_settings,
                    system_instruction=system_prompt 
                )
                
                # 4. Call generate_content_async on the model instance
                response = await model.generate_content_async(parts, stream=True)

                response_text = ""
                async for chunk in response: 
                    if hasattr(chunk, 'text') and chunk.text is not None:
                        response_text += chunk.text
                
                print(f"[Gemini API Raw Response]:\n{response_text}\n")

                validation_result = self._validate_response(response_text, val_text, index, job_type, required_headers)
                
                if validation_result[0] == "Error" and retries < max_retries - 1:
                    if job_type == "Metadata":
                        # Adjust temp in a new config object for the next retry
                        new_temp = min(0.9, float(current_generation_config.temperature if current_generation_config.temperature is not None else temp) + ((retries + 1) * 0.1))
                        current_generation_config = types.GenerationConfig(
                            temperature=new_temp,
                            top_p=current_generation_config.top_p,
                            top_k=current_generation_config.top_k,
                            max_output_tokens=current_generation_config.max_output_tokens
                        )
                    retries += 1
                    await asyncio.sleep(1 * (1.5 ** retries))
                    continue
                
                return validation_result

            except Exception as e:
                if "API_KEY_INVALID" in str(e) or "API key not valid" in str(e):
                     self.log_error(f"Gemini API Key is invalid or not configured properly for index {index}.", f"{str(e)}")
                     return "Error: Invalid API Key", index

                print(f"[Gemini API Exception]: {e}") # Keep this for immediate feedback
                self.log_error(f"Gemini API Error with {engine} for index {index}", f"{str(e)}")
                retries += 1
                if retries == max_retries:
                    return "Error", index
                await asyncio.sleep(1 * (1.5 ** retries))
        return "Error", index # Should be unreachable if loop exits normally
    
    async def handle_claude_call(self, system_prompt, user_prompt, temp, image_data, 
                                text_to_process, val_text, engine, index, 
                                is_base64=True, formatting_function=False, api_timeout=120.0,
                                job_type=None, required_headers=None):
        """Handle API calls to Anthropic Claude models"""
        async with AsyncAnthropic(api_key=self.anthropic_api_key, 
                                max_retries=0, timeout=api_timeout) as client:
            populated_user_prompt = user_prompt if formatting_function else user_prompt.format(text_to_process=text_to_process)

            # Set max_tokens based on job type or prompt contents
            if job_type == "Metadata":
                max_tokens = 2000
            elif "Pagination:" in user_prompt.lower() or "Split Before:" in user_prompt:
                max_tokens = 200
            elif "extract information" in user_prompt.lower():
                max_tokens = 1500
            else:
                max_tokens = 1200

            try:
                # Prepare message content with images if present
                content = []
                
                if isinstance(image_data, list) and image_data:
                    for img, label in image_data:
                        if label:
                            content.append({"type": "text", "text": label})
                        # Ensure img is a valid base64 string
                        if isinstance(img, bytes):
                            img = base64.b64encode(img).decode('utf-8')
                        
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img
                            }
                        })
                elif isinstance(image_data, str):
                    content = [
                        {"type": "text", "text": "Document Image:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        }
                    ]

                # Add the user prompt at the end
                if populated_user_prompt.strip():
                    content.append({"type": "text", "text": populated_user_prompt.strip()})

                max_retries = 5 if job_type == "Metadata" else 3
                retries = 0
                current_temp = temp
                current_max_tokens = max_tokens
                
                while retries < max_retries:
                    try:
                        message = await client.messages.create(
                            max_tokens=current_max_tokens,
                            messages=[{"role": "user", "content": content}],
                            system=system_prompt,
                            model=engine,
                            temperature=current_temp,
                            timeout=api_timeout
                        )
                        
                        response = message.content[0].text
                        validation_result = self._validate_response(response, val_text, index, job_type, required_headers)
                        
                        if validation_result[0] == "Error" and retries < max_retries - 1:
                            if job_type == "Metadata":
                                current_temp = min(0.9, float(current_temp) + (retries * 0.1))
                                if retries >= 2:
                                    current_max_tokens = min(4000, current_max_tokens + 500)
                            
                            retries += 1
                            await asyncio.sleep(1 * (1.5 ** retries))
                            continue
                        
                        return validation_result

                    except (anthropic.APITimeoutError, anthropic.APIError) as e:
                        self.log_error(f"Claude API Error with {engine} for index {index}", f"{str(e)}")
                        retries += 1
                        if retries == max_retries:
                            return "Error", index
                        await asyncio.sleep(1 * (1.5 ** retries))
                        
            except Exception as e:
                self.log_error(f"Error preparing Claude content for index {index}", f"{str(e)}")
                return "Error", index
                
    def _validate_response(self, response, val_text, index, job_type=None, required_headers=None):
        """
        Validates API response against requirements
        
        Args:
            response: The response text from the API
            val_text: Optional validation text to look for
            index: Document index
            job_type: Type of job being processed
            required_headers: List of required headers for metadata validation
            
        Returns:
            Tuple of (processed_response, index) or ("Error", index)
        """
        # First check if we have a valid response
        if not response:
            self.log_error(f"Empty API response for index {index}", f"job_type: {job_type}")
            return "Error", index
            
        # If no validation text is needed, return the full response
        if not val_text or val_text == "None":
            return response, index
            
        # Check if validation text exists in response
        try:
            if val_text in response:
                processed_response = response.split(val_text, 1)[1].strip()
                
                # Special validation for Metadata responses
                if job_type == "Metadata" and required_headers:
                    # Check if all required headers are present and have content
                    missing_headers = []
                    has_content = False
                    
                    for header in required_headers:
                        header_pattern = f"{header}:"
                        if header_pattern not in processed_response:
                            missing_headers.append(header)
                            continue
                            
                        # Check if header has actual content
                        try:
                            split_parts = processed_response.split(header_pattern, 1)
                            if len(split_parts) > 1:
                                header_content = split_parts[1].strip()
                                
                                # Find end of this header's content (next header or end of text)
                                next_header_pos = float('inf')
                                for next_header in required_headers:
                                    next_pattern = f"\n{next_header}:"
                                    pos = header_content.find(next_pattern)
                                    if pos != -1 and pos < next_header_pos:
                                        next_header_pos = pos
                                
                                if next_header_pos != float('inf'):
                                    header_content = header_content[:next_header_pos].strip()
                                
                                # Check if there's meaningful content
                                if header_content and not header_content.isspace():
                                    has_content = True
                        except Exception as e:
                            self.log_error(f"Error checking content for header {header}", f"{str(e)}")
                    
                    # Request error (retry) if headers are missing or all headers are empty
                    if missing_headers:
                        self.log_error(f"Missing required headers in metadata response", f"Missing: {missing_headers}, index: {index}")
                        return "Error", index
                    
                    if not has_content:
                        self.log_error(f"All metadata headers are empty", f"index: {index}, job_type: {job_type}")
                        return "Error", index
                
                return processed_response, index
            else:
                self.log_error(f"Validation text not found in response", f"val_text: {val_text}, index: {index}")
        except TypeError:
            self.log_error(f"Validation error - Response type mismatch", 
                        f"Response: {type(response)}, Val_text: {type(val_text)}")
            return "Error", index
            
        return "Error", index

    def prepare_image_data(self, image_data, engine, is_base64=True):
        """
        Prepare image data in the format required by the specified engine
        
        Args:
            image_data: Image path(s) or data
            engine: The AI model engine being used
            is_base64: Whether to encode as base64
            
        Returns:
            Processed image data ready for the API
        """
        if not image_data:
            return None

        # For Gemini, return the file paths directly
        if "gemini" in engine.lower():
            return image_data

        needs_base64 = is_base64 and ("gpt" in engine.lower() or "o1" in engine.lower() 
                                     or "o3" in engine.lower() or "claude" in engine.lower())
        
        # Handle single image case
        if isinstance(image_data, str):
            return self.encode_image(image_data) if needs_base64 else image_data

        # Handle multiple images case
        processed_data = []
        for img_path, label in image_data:
            if needs_base64:
                encoded_image = self.encode_image(img_path)
                if encoded_image:
                    processed_data.append((encoded_image, label))
            else:
                processed_data.append((img_path, label))

        return processed_data

    def encode_image(self, image_path):
        """Convert image file to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.log_error(f"Error encoding image", f"Path: {image_path}, Error: {str(e)}")
            return None
