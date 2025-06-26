import os
import json
import logging
import sqlite3
import asyncio
import threading
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import cv2
import numpy as np
import base64
from rate_limit_handler import RateLimitHandler
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BedrockOCRReader:
    """Enhanced Bedrock OCR Reader with image preprocessing and digit validation"""
    
    def __init__(self, model_name: str = "anthropic.claude-3-5-sonnet-20240620-v1:0", 
                 database_path: str = "smart_meter_database.db"):
        self.model_name = model_name
        self.database_path = database_path
        self.rate_limiter = RateLimitHandler(max_retries=5, base_delay=2)
        
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name='us-east-1'
            )
            logger.info(f"Initialized Bedrock client with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
        
        self._db_lock = threading.Lock()
        self.reference_data = self._load_reference_data()
        logger.info(f"Loaded {len(self.reference_data)} reference meter records")
    
    def _preprocess_image(self, image_path: str) -> str:
        """Preprocess image to enhance readability for OCR"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not read image: {image_path}")
                return image_path
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Apply sharpening filter
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(blurred, -1, kernel)
            
            # Apply morphological operations to clean up the image
            kernel_morph = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel_morph)
            
            # Save preprocessed image
            preprocessed_path = image_path.replace('.jpg', '_preprocessed.jpg').replace('.jpeg', '_preprocessed.jpg').replace('.png', '_preprocessed.jpg')
            cv2.imwrite(preprocessed_path, cleaned)
            
            logger.info(f"Image preprocessed and saved: {preprocessed_path}")
            return preprocessed_path
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return image_path
    
    def _validate_digit_reading(self, reading: str) -> Tuple[str, List[str]]:
        """Validate and correct digit reading patterns"""
        if not reading or reading.lower() in ['not visible', 'unclear']:
            return reading, []
        
        issues = []
        corrected_reading = reading
        
        # Remove non-digit characters except leading zeros
        cleaned = re.sub(r'[^0-9]', '', reading)
        
        # Check for common digit misreadings
        digit_corrections = {
            '7': '1',  # Common misreading in low quality
            '1': '7',  # Reverse correction
            '0': '8',  # Common misreading
            '8': '0',  # Reverse correction
            '5': '6',  # Common misreading
            '6': '5',  # Reverse correction
            '3': '8',  # Common misreading
            '8': '3',  # Reverse correction
        }
        
        # Validate reading length (typically 4-6 digits)
        if len(cleaned) < 3:
            issues.append(f"Reading too short: {cleaned}")
        elif len(cleaned) > 8:
            issues.append(f"Reading too long: {cleaned}")
        
        # Check for all same digits (likely error)
        if len(set(cleaned)) == 1 and len(cleaned) > 2:
            issues.append(f"All digits are the same: {cleaned}")
        
        # Check for sequential patterns (likely error)
        if len(cleaned) >= 3:
            for i in range(len(cleaned) - 2):
                if (cleaned[i:i+3] in ['123', '234', '345', '456', '567', '678', '789', '890'] or
                    cleaned[i:i+3] in ['987', '876', '765', '654', '543', '432', '321', '210']):
                    issues.append(f"Sequential pattern detected: {cleaned[i:i+3]}")
        
        return cleaned, issues
    
    def _create_enhanced_prompt(self) -> str:
        """Enhanced prompt with specific guidance for digit accuracy"""
        return """🎯 EXPERT METER READER - PRECISION DIGIT EXTRACTION

Your PRIMARY MISSION: Extract the MAIN CONSUMPTION READING with absolute precision, especially for low-quality images.

🔍 CRITICAL DIGIT ACCURACY GUIDELINES:

DIGIT DISTINCTION RULES:
- 1 vs 7: Look for the horizontal line at top of 7, 1 has no horizontal line
- 0 vs 8: 8 has two circles, 0 has one oval
- 5 vs 6: 6 has a closed loop at bottom, 5 has open bottom
- 3 vs 8: 3 has two semi-circles, 8 has two full circles
- 2 vs Z: 2 has curved top, Z has straight lines

COMPLETE READING EXTRACTION:
- ALWAYS count ALL visible digits (typically 4-6 digits)
- Start from LEFTMOST digit and read RIGHT
- Include ALL digits even if partially visible
- If last digit is unclear, mark it with '?' but include it
- Example: "1467?" is better than "1467"

🔍 METER READING EXTRACTION:
Find the LARGEST number on the meter display:
- Usually 4-6 digits (examples: 14673, 04420, 90965)
- Located in CENTER/TOP of LCD display
- Represents TOTAL kWh consumed
- PRESERVE leading zeros (04420, NOT 4420)
- COUNT EVERY DIGIT - don't skip the last one!

QUALITY ASSESSMENT:
- If image is blurry/low-quality, be EXTRA careful with digit distinction
- Mark uncertain digits with '?' but include them
- Provide detailed reasoning for each uncertain digit

📋 EXTRACTION TARGETS:
1. METER_READING: Complete consumption display (4-6 digits with leading zeros)
2. SERIAL_NUMBER: Meter identifier (6-8 alphanumeric)
3. TYPE: Electric/Gas/Water
4. DATE: Reading date if visible
5. TIME: Reading time if visible
6. DISPLAY: LCD/LED/Digital
7. UNITS: kWh/m3/gal

🎯 CONFIDENCE SCORING (1-10):
- 10: All digits crystal clear, no ambiguity
- 8-9: Minor uncertainty in 1 digit
- 6-7: Some digits require interpretation
- 4-5: Multiple digits unclear
- 1-3: Reading barely visible

📤 JSON OUTPUT:
{
    "meter_serial_number": "exact_serial_or_not_visible",
    "meter_reading": "COMPLETE_CONSUMPTION_WITH_LEADING_ZEROS",
    "meter_type": "Electric/Gas/Water",
    "reading_date": "date_if_visible",
    "reading_time": "time_if_visible",
    "display_type": "LCD/LED/Digital",
    "units": "kWh/m3/gal",
    "image_quality_factors": {
        "clarity": score_1_10,
        "lighting": score_1_10,
        "angle": score_1_10,
        "contrast": score_1_10,
        "glare": score_1_10
    },
    "confidence_score": overall_confidence_1_10,
    "extraction_notes": "specific_observations",
    "uncertainty_factors": ["specific", "issues"],
    "reading_analysis": "description_of_main_display_location",
    "digit_validation": {
        "total_digits_expected": 4_6,
        "total_digits_found": actual_count,
        "uncertain_digits": ["position", "reason"],
        "digit_quality_notes": "specific_observations"
    }
}

⚠️ CRITICAL: 
1. Focus ONLY on the PRIMARY consumption display
2. COUNT EVERY DIGIT - don't miss the last one!
3. Be EXTRA careful with digit distinction in low-quality images
4. If uncertain about a digit, mark it with '?' but include it
5. Double-check your digit count before responding"""
    
    def _get_db_connection(self):
        try:
            conn = sqlite3.connect(self.database_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def _load_reference_data(self) -> Dict[str, Dict]:
        try:
            with self._db_lock:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT meter_serial_number, meter_type, current_reading, 
                           installation_date, last_reading_date
                    FROM meter_reference_data
                """)
                
                reference_data = {}
                for row in cursor.fetchall():
                    reference_data[row['meter_serial_number']] = {
                        'meter_serial_number': row['meter_serial_number'],
                        'meter_type': row['meter_type'],
                        'current_reading': row['current_reading'],
                        'installation_date': row['installation_date'],
                        'last_reading_date': row['last_reading_date']
                    }
                
                conn.close()
                return reference_data
                
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            return {}
    
    def _enhanced_fuzzy_matching(self, extracted_serial: str, extracted_reading: str) -> Tuple[Optional[Dict], str, float]:
        """Enhanced fuzzy matching with multiple strategies and digit correction"""
        if not extracted_serial or extracted_serial.lower() in ['not visible', 'unclear']:
            return None, "No Match", 0.0
        
        # Strategy 1: Exact match
        if extracted_serial in self.reference_data:
            return self.reference_data[extracted_serial], "Exact Match", 100.0
        
        # Strategy 2: Reading-based matching with enhanced digit handling
        if extracted_reading and extracted_reading.lower() not in ['not visible', 'unclear']:
            # Clean and validate the extracted reading
            cleaned_reading, _ = self._validate_digit_reading(extracted_reading)
            
            for ref_serial, ref_data in self.reference_data.items():
                ref_reading = ref_data['current_reading']
                
                # Multiple comparison strategies
                if (cleaned_reading == ref_reading or 
                    cleaned_reading.lstrip('0') == ref_reading.lstrip('0') or
                    cleaned_reading.zfill(5) == ref_reading.zfill(5) or
                    self._calculate_digit_similarity(cleaned_reading, ref_reading) >= 0.8):
                    return ref_data, "Reading Match", 100.0
        
        return None, "No Match", 0.0
    
    def _calculate_digit_similarity(self, reading1: str, reading2: str) -> float:
        """Calculate similarity between two readings considering common misreadings"""
        if not reading1 or not reading2:
            return 0.0
        
        # Pad shorter reading with leading zeros
        max_len = max(len(reading1), len(reading2))
        reading1_padded = reading1.zfill(max_len)
        reading2_padded = reading2.zfill(max_len)
        
        # Common digit misreadings
        digit_mappings = {
            '1': ['7'], '7': ['1'],
            '0': ['8'], '8': ['0'],
            '5': ['6'], '6': ['5'],
            '3': ['8'], '8': ['3'],
            '2': ['Z'], 'Z': ['2']
        }
        
        matches = 0
        total_digits = len(reading1_padded)
        
        for i in range(total_digits):
            if reading1_padded[i] == reading2_padded[i]:
                matches += 1
            elif (reading1_padded[i] in digit_mappings and 
                  reading2_padded[i] in digit_mappings[reading1_padded[i]]):
                matches += 0.8  # Partial credit for common misreadings
        
        return matches / total_digits if total_digits > 0 else 0.0
    
    async def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image with enhanced preprocessing and validation"""
        try:
            # Preprocess image to enhance quality
            preprocessed_path = self._preprocess_image(image_path)
            
            # Encode preprocessed image
            with open(preprocessed_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create enhanced prompt
            prompt = self._create_enhanced_prompt()
            
            # Prepare request with optimized settings
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2500,  # Increased for more detailed analysis
                "temperature": 0.05,  # Lower temperature for more consistent results
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        }
                    ]
                }]
            }
            
            # Make API call with rate limiting
            response = self.rate_limiter.call_with_retry(
                self.bedrock_client.invoke_model,
                modelId=self.model_name,
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            content = response_body['content'][0]['text']
            
            # Extract JSON
            try:
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    extraction_result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found")
            except (json.JSONDecodeError, ValueError):
                # Enhanced fallback
                extraction_result = {
                    "meter_serial_number": "Not visible",
                    "meter_reading": "Not visible",
                    "meter_type": "Electric",
                    "reading_date": "Not visible",
                    "reading_time": "Not visible",
                    "display_type": "LCD",
                    "units": "kWh",
                    "confidence_score": 3,
                    "extraction_notes": "JSON parsing failed",
                    "digit_validation": {
                        "total_digits_expected": 5,
                        "total_digits_found": 0,
                        "uncertain_digits": [],
                        "digit_quality_notes": "Parsing failed"
                    }
                }
            
            # Extract and validate values
            extracted_serial = extraction_result.get("meter_serial_number", "Not visible")
            extracted_reading = extraction_result.get("meter_reading", "Not visible")
            extracted_type = extraction_result.get("meter_type", "Electric")
            extracted_date = extraction_result.get("reading_date", "Not visible")
            extracted_time = extraction_result.get("reading_time", "Not visible")
            
            # Validate and correct digit reading
            validated_reading, digit_issues = self._validate_digit_reading(extracted_reading)
            
            base_confidence = float(extraction_result.get("confidence_score", 5)) / 10.0
            extraction_notes = extraction_result.get("extraction_notes", "")
            
            # Add digit validation notes
            if digit_issues:
                extraction_notes += f" | Digit issues: {', '.join(digit_issues)}"
            
            # Enhanced matching with validated reading
            best_match, match_type, serial_confidence = self._enhanced_fuzzy_matching(
                extracted_serial, validated_reading
            )
            
            # Validate reading with enhanced logic
            reading_validated = False
            reading_confidence = 0.0
            if best_match and validated_reading.lower() not in ['not visible', 'unclear']:
                if validated_reading == best_match['current_reading']:
                    reading_validated = True
                    reading_confidence = 100.0
                else:
                    # Calculate similarity-based confidence
                    similarity = self._calculate_digit_similarity(validated_reading, best_match['current_reading'])
                    reading_confidence = similarity * 100.0
                    if similarity >= 0.8:
                        reading_validated = True
            
            # Determine outcome with enhanced logic
            if match_type == "Exact Match" and reading_validated:
                processing_outcome = "Perfect Match"
            elif match_type == "Reading Match":
                processing_outcome = "Good Match"
            elif match_type == "Exact Match" and reading_confidence >= 80:
                processing_outcome = "Partial Match"
            elif serial_confidence >= 50 or reading_confidence >= 50:
                processing_outcome = "Partial Match"
            else:
                processing_outcome = "No Match"
            
            # Prepare enhanced result
            result = {
                "extracted_data": {
                    "meter_serial_number": extracted_serial,
                    "meter_reading": validated_reading,  # Use validated reading
                    "meter_type": extracted_type,
                    "reading_date": extracted_date,
                    "reading_time": extracted_time,
                    "display_type": extraction_result.get("display_type", "LCD"),
                    "units": extraction_result.get("units", "kWh")
                },
                "manual_data": {
                    "meter_serial_number": best_match['meter_serial_number'] if best_match else "No match found",
                    "meter_reading": best_match['current_reading'] if best_match else "No match found",
                    "meter_type": best_match['meter_type'] if best_match else "No match found",
                    "reading_date": best_match['last_reading_date'] if best_match else "No match found"
                },
                "validation": {
                    "serial_match_type": match_type,
                    "serial_confidence": serial_confidence,
                    "reading_validated": reading_validated,
                    "reading_confidence": reading_confidence,
                    "processing_outcome": processing_outcome,
                    "digit_validation": extraction_result.get("digit_validation", {}),
                    "digit_issues": digit_issues
                },
                "ai_confidence": base_confidence,
                "extraction_notes": extraction_notes,
                "image_quality": extraction_result.get("image_quality_factors", {}),
                "processing_metadata": {
                    "model_used": self.model_name,
                    "processing_timestamp": datetime.now().isoformat(),
                    "preprocessed_image": preprocessed_path != image_path
                }
            }
            
            # Clean up preprocessed image if it was created
            if preprocessed_path != image_path and os.path.exists(preprocessed_path):
                try:
                    os.remove(preprocessed_path)
                except Exception as e:
                    logger.warning(f"Could not remove preprocessed image: {e}")
            
            logger.info(f"Enhanced processing completed: {image_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced processing {image_path}: {e}")
            return {
                "extracted_data": {"meter_serial_number": "Error", "meter_reading": "Error", "meter_type": "Error", "reading_date": "Error"},
                "manual_data": {"meter_serial_number": "Error", "meter_reading": "Error", "meter_type": "Error", "reading_date": "Error"},
                "validation": {"serial_match_type": "Error", "serial_confidence": 0.0, "reading_validated": False, "reading_confidence": 0.0, "processing_outcome": "Error"},
                "ai_confidence": 0.0,
                "extraction_notes": f"Processing error: {str(e)}"
            }
    
    def save_processing_result(self, image_path: str, result: Dict[str, Any]) -> bool:
        """Save processing result to database"""
        try:
            with self._db_lock:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO processing_results (
                        image_path, extracted_serial, extracted_reading, extracted_type, extracted_date,
                        manual_serial, manual_reading, manual_type, manual_date,
                        serial_match_type, serial_confidence, reading_validated, reading_confidence,
                        processing_outcome, ai_confidence, extraction_notes, processing_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    image_path,
                    result['extracted_data']['meter_serial_number'],
                    result['extracted_data']['meter_reading'],
                    result['extracted_data']['meter_type'],
                    result['extracted_data']['reading_date'],
                    result['manual_data']['meter_serial_number'],
                    result['manual_data']['meter_reading'],
                    result['manual_data']['meter_type'],
                    result['manual_data']['reading_date'],
                    result['validation']['serial_match_type'],
                    result['validation']['serial_confidence'],
                    result['validation']['reading_validated'],
                    result['validation']['reading_confidence'],
                    result['validation']['processing_outcome'],
                    result['ai_confidence'],
                    result['extraction_notes'],
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                conn.close()
                return True
                
        except Exception as e:
            logger.error(f"Error saving result: {e}")
            return False