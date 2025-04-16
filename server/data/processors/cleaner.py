"""
Data cleaning utilities for preprocessing data.
"""
import re
import unicodedata
from typing import Optional, Union, Dict, Any, List, Tuple
from datetime import datetime


class DataCleaner:
    """
    Cleans and normalizes data from various sources.
    """
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and normalizing.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Normalize Unicode characters
        text = unicodedata.normalize("NFKC", text)
        
        # Replace multiple spaces with a single space
        cleaned = re.sub(r'\s+', ' ', text)
        # Trim whitespace
        cleaned = cleaned.strip()
        # Remove non-printable characters
        cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned)
        
        return cleaned
    
    def extract_numeric_value(self, value: Union[str, int, float]) -> Optional[float]:
        """
        Extract numeric value from a string or return the numeric value itself.
        
        Args:
            value: Value to extract numeric part from
            
        Returns:
            Numeric value or None if no valid number found
        """
        if value is None:
            return None
            
        # If already a number, return it
        if isinstance(value, (int, float)):
            return float(value)
            
        if not isinstance(value, str):
            return None
            
        # Remove all non-numeric characters except decimal point and negative sign
        cleaned = value.strip()
        
        # Try to match a numeric pattern
        # This handles:
        # - Integers: 123
        # - Decimals: 123.45
        # - Negative numbers: -123.45
        # - Numbers with commas as thousand separators: 1,234.56
        
        # First, remove thousand separators if present
        cleaned = cleaned.replace(',', '')
        
        # Match numeric pattern
        matches = re.search(r'^-?\d+\.?\d*', cleaned)
        
        if matches:
            try:
                return float(matches.group(0))
            except ValueError:
                return None
        
        # If no match found, try common patterns like "123 pts" or "$123.45"
        matches = re.search(r'(-?\d+\.?\d*)', cleaned)
        if matches:
            try:
                return float(matches.group(1))
            except ValueError:
                return None
        
        return None
    
    def convert_height_to_cm(self, height_str: str) -> Optional[float]:
        """
        Convert height from feet/inches format to centimeters.
        
        Args:
            height_str: Height string in format like "6'2" or "6-2" or "6ft 2in"
            
        Returns:
            Height in centimeters or None if parsing fails
        """
        if not height_str or not isinstance(height_str, str):
            return None
            
        # Match various height formats
        # 6'2, 6-2, 6ft 2in, 6 feet 2 inches, etc.
        pattern = r'(\d+)[\'\-\s]?(?:ft|feet|foot|)[\.\s\'\-]*(\d+)?(?:in|inch|inches|\")?'
        match = re.search(pattern, height_str)
        
        if match:
            feet = int(match.group(1))
            inches = int(match.group(2)) if match.group(2) else 0
            
            # Convert to cm (1 foot = 30.48 cm, 1 inch = 2.54 cm)
            return round((feet * 30.48) + (inches * 2.54), 2)
        
        # Check if it's already in cm
        cm_pattern = r'(\d+\.?\d*)[\s]*(?:cm|centimeters|centimetres)'
        cm_match = re.search(cm_pattern, height_str)
        
        if cm_match:
            return float(cm_match.group(1))
        
        # Just try to extract any number
        numeric_value = self.extract_numeric_value(height_str)
        
        # If it's a small number (like 6.2), assume it's in feet and convert
        if numeric_value and numeric_value < 10:
            feet = int(numeric_value)
            inches = int((numeric_value - feet) * 10)
            return round((feet * 30.48) + (inches * 2.54), 2)
        
        return numeric_value
    
    def convert_weight_to_kg(self, weight_str: str) -> Optional[float]:
        """
        Convert weight from pounds to kilograms.
        
        Args:
            weight_str: Weight string in format like "195 lbs" or "195 pounds"
            
        Returns:
            Weight in kilograms or None if parsing fails
        """
        if not weight_str or not isinstance(weight_str, str):
            return None
        
        # Check if it's already in kg
        kg_pattern = r'(\d+\.?\d*)[\s]*(?:kg|kilograms|kilos)'
        kg_match = re.search(kg_pattern, weight_str)
        
        if kg_match:
            return float(kg_match.group(1))
        
        # Check if it's in pounds
        lb_pattern = r'(\d+\.?\d*)[\s]*(?:lbs|pounds|lb)'
        lb_match = re.search(lb_pattern, weight_str)
        
        if lb_match:
            # Convert pounds to kg (1 pound = 0.453592 kg)
            return round(float(lb_match.group(1)) * 0.453592, 2)
        
        # Just try to extract any number
        numeric_value = self.extract_numeric_value(weight_str)
        
        # If it's a large number (like 195), assume it's in pounds and convert
        if numeric_value and numeric_value > 50:
            return round(numeric_value * 0.453592, 2)
        
        return numeric_value
    
    def clean_date_string(self, date_str: str) -> Optional[str]:
        """
        Clean and normalize date string to YYYY-MM-DD format.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            Cleaned date string in YYYY-MM-DD format or None if parsing fails
        """
        if not date_str:
            return None
        
        # Try various date formats
        formats = [
            "%Y-%m-%d",  # ISO format
            "%Y/%m/%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%B %d, %Y",  # January 1, 2021
            "%b %d, %Y",  # Jan 1, 2021
            "%d %B %Y",   # 1 January 2021
            "%d %b %Y",   # 1 Jan 2021
            "%Y.%m.%d",
            "%d.%m.%Y"
        ]
        
        for fmt in formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        return None
    
    def clean_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean all string values in a dictionary.
        
        Args:
            data: Dictionary to clean
            
        Returns:
            Dictionary with cleaned string values
        """
        if not data or not isinstance(data, dict):
            return {}
            
        cleaned_data = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                cleaned_data[key] = self.clean_text(value)
            elif isinstance(value, dict):
                cleaned_data[key] = self.clean_dict(value)
            elif isinstance(value, list):
                cleaned_data[key] = self.clean_list(value)
            else:
                cleaned_data[key] = value
                
        return cleaned_data
    
    def clean_list(self, data: List[Any]) -> List[Any]:
        """
        Clean all items in a list.
        
        Args:
            data: List to clean
            
        Returns:
            List with cleaned items
        """
        if not data or not isinstance(data, list):
            return []
            
        cleaned_data = []
        
        for item in data:
            if isinstance(item, str):
                cleaned_data.append(self.clean_text(item))
            elif isinstance(item, dict):
                cleaned_data.append(self.clean_dict(item))
            elif isinstance(item, list):
                cleaned_data.append(self.clean_list(item))
            else:
                cleaned_data.append(item)
                
        return cleaned_data
    
    def normalize_team_name(self, name: str) -> str:
        """
        Normalize a team name for consistent matching.
        
        Args:
            name: Team name
            
        Returns:
            Normalized team name
        """
        if not name:
            return ""
        
        # Clean the name first
        name = self.clean_text(name)
        
        # Convert to lowercase
        name = name.lower()
        
        # Remove common prefixes like "The"
        prefixes = ["the ", "los angeles ", "la "]
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
        
        # Remove common suffixes like "Basketball Team"
        suffixes = [" basketball team", " basketball", " team"]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        
        # Handle common abbreviations and variations
        name_map = {
            "sixers": "76ers",
            "blazers": "trail blazers",
            "wolves": "timberwolves",
            "cavs": "cavaliers",
            "mavs": "mavericks",
            "knicks": "knickerbockers"
        }
        
        # Apply mapping if name matches
        for abbr, full in name_map.items():
            if name == abbr:
                name = full
                break
        
        return name.strip()