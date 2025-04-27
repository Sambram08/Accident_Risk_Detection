from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import json
import logging
from .model_loader import predict_severity # Import the prediction function

# Setup logging
logger = logging.getLogger(__name__)

def plant_model(request):
    """Simple test view."""
    return HttpResponse("🌱 Plant Disease Detection API is Running!")

@csrf_exempt  # Disable CSRF for testing (use proper security in production)
def predict_disease(request):
    if request.method == 'GET':
        return JsonResponse({"message": "Send a POST request with numerical data to get predictions."})

    if request.method == 'POST':
        try:
            # Parse the incoming JSON data
            body_unicode = request.body.decode('utf-8')
            body_data = json.loads(body_unicode)

            # Expecting a key called 'input' containing a list of numbers
            if 'input' not in body_data:
                return JsonResponse({"error": "Missing 'input' field in request."}, status=400)

            input_data = body_data['input']

            if not isinstance(input_data, list):
                return JsonResponse({"error": "'input' must be a list of numerical values."}, status=400)

            # Call the model prediction function
            result = predict_severity(input_data)

            return JsonResponse({"prediction": result})

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format."}, status=400)

        except Exception as e:
            logger.error(f"❌ Error during prediction: {e}")
            return JsonResponse({"error": f"Internal Server Error: {str(e)}"}, status=500)

    return JsonResponse({"error": "Invalid request method."}, status=405)