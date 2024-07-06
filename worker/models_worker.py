import json
from typing import Dict, Any

from sqlalchemy import select
from worker.models.yolo_detection import Model as YoloModel
from api.db_model import TransactionHistory, get_session, TransactionStatusEnum, UploadedFile
from sqlalchemy.exc import NoResultFound
from worker.models.yolo_detection.utils import FileProcessor
from api.s3 import s3, S3Client
import traceback

yolo_model = YoloModel()
file_proc = FileProcessor()


async def analyze_uploaded_file(
    ctx: Dict[str, Any],
    uploaded_file_id : str,
    uploaded_file_name: str,
    **kwargs: Any,
):
    async for session in get_session():
        try:
            
            job_id = ctx.get("job_id", None)
            if not job_id:
                raise Exception("Something is wrong. job_id is None")

            transaction = await session.execute(
                select(TransactionHistory).filter_by(job_id=job_id)
            )
            transaction = transaction.scalar()
            
            doc = await session.execute(
                select(UploadedFile).filter_by(id=uploaded_file_id)
            )
            doc = doc.scalar()
            
            data = await file_proc.process_file(uploaded_file_name, uploaded_file_id)
            # Предикт yolo
            result = await yolo_model.predict(data)
            # result = model.predict(data)
            #result = "Test results"
            if not result:
                raise Exception(f"Something is wrong. Try again later: {result}")

            transaction.status = TransactionStatusEnum.SUCCESS

            json_data = json.dumps(result)
            transaction.result = json_data
            await session.commit()
            return result
        except Exception as e:
            transaction.status = TransactionStatusEnum.FAILURE
            transaction.err_reason = str(e)
            doc.verified = False
            doc.cancellation_reason = f"File processing error: {e}"
            await session.commit()
            traceback.print_exc()
            return json.dumps({"data": uploaded_file_id, "result": str(e)})
