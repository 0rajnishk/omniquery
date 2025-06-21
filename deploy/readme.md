<!-- uvicorn main:app --host 127.0.0.1 --port 8000 --reload -->



gcloud builds submit --tag gcr.io/d-meeting-rooms/adk-trial



gcloud config set account comvikram.desicrew@gmail.com 


gcloud run deploy adk-trial --image gcr.io/d-meeting-rooms/adk-trial --platform managed --region asia-south1 --allow-unauthenticated --service-account=meeting-room-app-sa@d-meeting-rooms.iam.gserviceaccount.com



