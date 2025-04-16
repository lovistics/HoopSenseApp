# server/data/processors/validator.py
"""
Data validation utilities for ensuring data quality.
"""
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Callable, Set
import re


class DataValidator:
    """
    Validates data before processing or saving.
    """
    
    def validate_required_fields(
        self, 
        data: Dict[str, Any], 
        required_fields: List[str]
    ) -> Dict[str, List[str]]:
        """
        Validate that all required fields are present and not empty.
        
        Args:
            data: Data to validate
            required_fields: List of required field names
            
        Returns:
            Dictionary of errors by field name
        """
        errors = {}
        
        for field in required_fields:
            if field not in data:
                errors[field] = ["Field is required"]
            elif data[field] is None:
                errors[field] = ["Field cannot be null"]
            elif isinstance(data[field], str) and not data[field].strip():
                errors[field] = ["Field cannot be empty"]
                
        return errors
    
    def validate_field_type(
        self, 
        data: Dict[str, Any], 
        field: str, 
        expected_type: Union[type, List[type]]
    ) -> Optional[str]:
        """
        Validate that a field is of the expected type.
        
        Args:
            data: Data to validate
            field: Field name to check
            expected_type: Expected type or list of types
            
        Returns:
            Error message if validation fails, None otherwise
        """
        if field not in data or data[field] is None:
            return None
            
        if isinstance(expected_type, list):
            if not any(isinstance(data[field], t) for t in expected_type):
                return f"Field must be one of types: {', '.join(t.__name__ for t in expected_type)}"
        elif not isinstance(data[field], expected_type):
            return f"Field must be of type {expected_type.__name__}"
            
        return None
    
    def validate_field_format(
        self, 
        data: Dict[str, Any], 
        field: str, 
        validator_func: Callable[[Any], bool], 
        error_message: str
    ) -> Optional[str]:
        """
        Validate that a field matches a format using a validator function.
        
        Args:
            data: Data to validate
            field: Field name to check
            validator_func: Function that returns True if valid, False otherwise
            error_message: Error message to return if validation fails
            
        Returns:
            Error message if validation fails, None otherwise
        """
        if field not in data or data[field] is None:
            return None
            
        if not validator_func(data[field]):
            return error_message
            
        return None
    
    def validate_date_format(
        self, 
        data: Dict[str, Any], 
        field: str,
        format_str: str = "%Y-%m-%d"
    ) -> Optional[str]:
        """
        Validate that a field is a valid date string.
        
        Args:
            data: Data to validate
            field: Field name to check
            format_str: Expected date format string
            
        Returns:
            Error message if validation fails, None otherwise
        """
        if field not in data or data[field] is None:
            return None
            
        if not isinstance(data[field], str):
            return "Field must be a string date"
            
        try:
            datetime.strptime(data[field], format_str)
            return None
        except ValueError:
            return f"Field must be a valid date in format {format_str}"
    
    def validate_numeric_range(
        self, 
        data: Dict[str, Any], 
        field: str,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None
    ) -> Optional[str]:
        """
        Validate that a numeric field is within a range.
        
        Args:
            data: Data to validate
            field: Field name to check
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)
            
        Returns:
            Error message if validation fails, None otherwise
        """
        if field not in data or data[field] is None:
            return None
            
        if not isinstance(data[field], (int, float)):
            return "Field must be a number"
            
        value = data[field]
        
        if min_value is not None and value < min_value:
            return f"Field must be at least {min_value}"
            
        if max_value is not None and value > max_value:
            return f"Field must be at most {max_value}"
            
        return None
    
    def validate_enum(
        self, 
        data: Dict[str, Any], 
        field: str,
        allowed_values: List[Any]
    ) -> Optional[str]:
        """
        Validate that a field's value is one of a set of allowed values.
        
        Args:
            data: Data to validate
            field: Field name to check
            allowed_values: List of allowed values
            
        Returns:
            Error message if validation fails, None otherwise
        """
        if field not in data or data[field] is None:
            return None
            
        if data[field] not in allowed_values:
            values_str = ", ".join(str(v) for v in allowed_values)
            return f"Field must be one of: {values_str}"
            
        return None
    
    def validate_nested_dict(
        self, 
        data: Dict[str, Any], 
        field: str,
        validator_func: Callable[[Dict[str, Any]], Dict[str, List[str]]]
    ) -> Dict[str, List[str]]:
        """
        Validate a nested dictionary field.
        
        Args:
            data: Data to validate
            field: Field name to check
            validator_func: Function that validates the nested dictionary
            
        Returns:
            Dictionary of errors by field name
        """
        if field not in data or data[field] is None:
            return {}
            
        if not isinstance(data[field], dict):
            return {field: ["Field must be an object"]}
            
        nested_errors = validator_func(data[field])
        
        if not nested_errors:
            return {}
            
        # Prefix nested field names with parent field name
        prefixed_errors = {}
        for nested_field, errors in nested_errors.items():
            prefixed_errors[f"{field}.{nested_field}"] = errors
            
        return prefixed_errors
    
    def validate_list_items(
        self, 
        data: Dict[str, Any], 
        field: str,
        item_validator: Callable[[Any], Optional[str]]
    ) -> List[str]:
        """
        Validate each item in a list field.
        
        Args:
            data: Data to validate
            field: Field name to check
            item_validator: Function that validates each item
            
        Returns:
            List of error messages
        """
        if field not in data or data[field] is None:
            return []
            
        if not isinstance(data[field], list):
            return ["Field must be an array"]
            
        errors = []
        
        for i, item in enumerate(data[field]):
            error = item_validator(item)
            if error:
                errors.append(f"Item {i}: {error}")
                
        return errors
    
    def validate_email(self, email: str) -> bool:
        """
        Validate an email address.
        
        Args:
            email: Email address to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not email:
            return False
            
        # Basic pattern matching for email validation
        pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        return bool(re.match(pattern, email))
    
    def validate_url(self, url: str) -> bool:
        """
        Validate a URL.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not url:
            return False
            
        # Basic pattern matching for URL validation
        pattern = r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(/[-\w%!$&\'()*+,;=:]+)*(?:\?[-\w%!$&\'()*+,;=:/?]+)?(?:#[-\w%!$&\'()*+,;=:/?]+)?$'
        return bool(re.match(pattern, url))
    
    def has_duplicates(self, items: List[Any]) -> bool:
        """
        Check if a list contains duplicate values.
        
        Args:
            items: List to check
            
        Returns:
            True if duplicates found, False otherwise
        """
        seen = set()
        for item in items:
            if item in seen:
                return True
            seen.add(item)
        return False
    
    def validate_data_consistency(
        self, 
        data: Dict[str, Any], 
        rules: List[Callable[[Dict[str, Any]], Optional[str]]]
    ) -> List[str]:
        """
        Validate data against complex consistency rules.
        
        Args:
            data: Data to validate
            rules: List of rule functions that check consistency
            
        Returns:
            List of error messages
        """
        errors = []
        
        for rule in rules:
            error = rule(data)
            if error:
                errors.append(error)
                
        return errors