"""
MongoDB Atlas Connection Module
Handles async connections to MongoDB for face embeddings and attendance data.
"""

import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, ASCENDING, DESCENDING
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

import config

logger = logging.getLogger(__name__)

# MongoDB client and database references
client: Optional[AsyncIOMotorClient] = None
db = None


async def connect_mongodb():
    """Initialize MongoDB connection."""
    global client, db
    
    try:
        client = AsyncIOMotorClient(config.MONGODB_URI)
        db = client[config.MONGODB_DATABASE]
        
        # Test connection
        await client.admin.command('ping')
        logger.info("✅ Connected to MongoDB Atlas!")
        
        # Create indexes
        await create_indexes()
        
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        return False


async def close_mongodb():
    """Close MongoDB connection."""
    global client
    if client:
        client.close()
        logger.info("MongoDB connection closed")


async def create_indexes():
    """Create necessary indexes for efficient queries."""
    try:
        # Students collection indexes
        await db.students.create_indexes([
            IndexModel([("roll_number", ASCENDING)], unique=True),
            IndexModel([("is_active", ASCENDING)]),
            IndexModel([("name", ASCENDING)])
        ])
        
        # Attendance collection indexes
        await db.attendance.create_indexes([
            IndexModel([("student_id", ASCENDING), ("date", DESCENDING)]),
            IndexModel([("date", DESCENDING)]),
            IndexModel([("check_in_time", DESCENDING)])
        ])
        
        # Behavior logs collection indexes
        await db.behavior_logs.create_indexes([
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("date", DESCENDING)])
        ])
        
        logger.info("✅ MongoDB indexes created")
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")


# ===== STUDENT OPERATIONS =====

async def create_student(
    name: str, 
    roll_number: str, 
    email: str = None, 
    phone: str = None
) -> Dict[str, Any]:
    """Create a new student document."""
    student = {
        "name": name,
        "roll_number": roll_number,
        "email": email,
        "phone": phone,
        "face_embeddings": [],  # Will store up to 3 embeddings
        "face_image_paths": [],
        "is_active": True,
        "registered_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = await db.students.insert_one(student)
    student["_id"] = result.inserted_id
    return student


async def get_student_by_roll(roll_number: str) -> Optional[Dict]:
    """Get student by roll number."""
    return await db.students.find_one({"roll_number": roll_number})


async def get_student_by_id(student_id: str) -> Optional[Dict]:
    """Get student by ID."""
    from bson import ObjectId
    return await db.students.find_one({"_id": ObjectId(student_id)})


async def get_all_students(active_only: bool = True) -> List[Dict]:
    """Get all students."""
    query = {"is_active": True} if active_only else {}
    cursor = db.students.find(query).sort("name", ASCENDING)
    return await cursor.to_list(length=None)


async def delete_student(student_id: str) -> bool:
    """Soft-delete a student (set is_active=False)."""
    from bson import ObjectId
    result = await db.students.update_one(
        {"_id": ObjectId(student_id)},
        {"$set": {"is_active": False, "updated_at": datetime.utcnow()}}
    )
    return result.modified_count > 0


async def add_face_embedding(
    student_id: str, 
    embedding: np.ndarray, 
    image_path: str = None
) -> bool:
    """Add a face embedding to student (max 3)."""
    from bson import ObjectId
    
    # Convert numpy array to list for MongoDB storage
    if isinstance(embedding, np.ndarray):
        embedding_list = embedding.tolist()
    else:
        embedding_list = embedding # Already a list?

    if not embedding_list:
        return False

    result = await db.students.update_one(
        {"_id": ObjectId(student_id)},
        {
            "$push": {
                "face_embeddings": {"$each": [embedding_list], "$slice": -3},
                "face_image_paths": {"$each": [image_path] if image_path else [], "$slice": -3}
            },
            "$set": {"updated_at": datetime.utcnow()}
        }
    )
    return result.modified_count > 0


async def get_all_face_embeddings() -> List[Dict]:
    """Get all students with their face embeddings for recognition."""
    cursor = db.students.find(
        {"is_active": True, "face_embeddings": {"$ne": []}},
        {"name": 1, "roll_number": 1, "face_embeddings": 1}
    )
    students = await cursor.to_list(length=None)
    
    # Convert lists back to numpy arrays
    for student in students:
        student["face_embeddings"] = [
            np.array(emb) for emb in student["face_embeddings"]
        ]
    
    return students


# ===== ATTENDANCE OPERATIONS =====

async def mark_attendance(
    student_id: str,
    confidence: float = 1.0,
    auto_detected: bool = True,
    camera: str = "gate",
    status: str = "present",
    lecture: str = None # NEW
) -> Optional[Dict]:
    """Mark student attendance."""
    from bson import ObjectId
    
    today = date.today()
    today_str = today.isoformat()
    
    # Check duplicate
    query = {
        "student_id": ObjectId(student_id),
        "date": today_str
    }
    if lecture:
        query["lecture"] = lecture
        
    existing = await db.attendance.find_one(query)
    
    if existing:
        # If manual override, allow updating status
        if not auto_detected:
            await db.attendance.update_one(
                {"_id": existing["_id"]},
                {"$set": {"status": status, "check_in_time": datetime.utcnow()}}
            )
            existing["status"] = status
            return existing
        return None  # Already marked
    
    attendance = {
        "student_id": ObjectId(student_id),
        "date": today_str,
        "check_in_time": datetime.utcnow(),
        "status": status,
        "detection_confidence": confidence,
        "auto_detected": auto_detected,
        "camera": camera,
        "lecture": lecture # NEW
    }
    
    result = await db.attendance.insert_one(attendance)
    attendance["_id"] = result.inserted_id
    return attendance


# ===== LOGGING OPERATIONS =====

async def log_classroom_snapshot(data: Dict):
    """Log classroom behavior snapshot."""
    if db is None: return
    try:
        log_entry = {
            "timestamp": datetime.utcnow(),
            "student_count": data.get("student_count", 0),
            "stats": data.get("stats", {}),
            "behaviors": data.get("behaviors", []),
            "faces": data.get("faces", []), # Faces list with names/status
            "date": date.today().isoformat()
        }
        await db.classroom_logs.insert_one(log_entry)
    except Exception as e:
        logger.error(f"Error logging classroom snapshot: {e}")


async def log_security_alert(camera: str, image: str, description: str):
    """Log a security alert (e.g. Unknown person)."""
    if db is None: return
    try:
        alert = {
            "timestamp": datetime.utcnow(),
            "type": "unknown_person",
            "camera": camera,
            "description": description,
            "image": image, # Base64 image
            "date": date.today().isoformat()
        }
        await db.security_alerts.insert_one(alert)
    except Exception as e:
        logger.error(f"Error logging security alert: {e}")


async def get_today_attendance() -> List[Dict]:
    """Get today's attendance with student details."""
    today = date.today().isoformat()
    
    pipeline = [
        {"$match": {"date": today}},
        {"$lookup": {
            "from": "students",
            "localField": "student_id",
            "foreignField": "_id",
            "as": "student"
        }},
        {"$unwind": "$student"},
        {"$sort": {"check_in_time": -1}},
        {"$project": {
            "_id": 1,
            "student_id": 1,
            "name": "$student.name",
            "roll_number": "$student.roll_number",
            "check_in_time": 1,
            "status": 1,
            "detection_confidence": 1,
            "auto_detected": 1
        }}
    ]
    
    cursor = db.attendance.aggregate(pipeline)
    return await cursor.to_list(length=None)


async def get_attendance_stats() -> Dict:
    """Get attendance statistics for today."""
    today = date.today().isoformat()
    
    total_students = await db.students.count_documents({"is_active": True})
    present_today = await db.attendance.count_documents({"date": today, "status": "present"})
    
    return {
        "total_students": total_students,
        "present_today": present_today,
        "absent_today": total_students - present_today,
        "attendance_rate": round((present_today / total_students * 100) if total_students > 0 else 0, 1)
    }


async def get_attendance_trend(days: int = 7) -> List[Dict]:
    """Get attendance trend for the last N days."""
    from datetime import timedelta
    
    end_date = date.today()
    start_date = end_date - timedelta(days=days-1)
    
    pipeline = [
        {"$match": {
            "date": {"$gte": start_date.isoformat(), "$lte": end_date.isoformat()}
        }},
        {"$group": {
            "_id": "$date",
            "present": {"$sum": 1}
        }},
        {"$sort": {"_id": 1}}
    ]
    
    cursor = db.attendance.aggregate(pipeline)
    results = await cursor.to_list(length=None)
    
    total_students = await db.students.count_documents({"is_active": True})
    
    return [{
        "date": r["_id"],
        "present": r["present"],
        "absent": total_students - r["present"],
        "total": total_students
    } for r in results]


# ===== BEHAVIOR LOGS =====

async def log_behavior(behavior_stats: Dict):
    """Log behavior detection stats."""
    log_entry = {
        "timestamp": datetime.utcnow(),
        "date": date.today().isoformat(),
        "hour": datetime.now().hour,
        "stats": behavior_stats
    }
    await db.behavior_logs.insert_one(log_entry)


async def get_behavior_summary(date_str: str = None) -> Dict:
    """Get aggregated behavior summary for a date."""
    if not date_str:
        date_str = date.today().isoformat()
    
    pipeline = [
        {"$match": {"date": date_str}},
        {"$group": {
            "_id": None,
            "total_detections": {"$sum": 1},
            "studying": {"$avg": "$stats.studying"},
            "focused": {"$avg": "$stats.focused"},
            "distracted": {"$avg": "$stats.distracted"},
            "sleeping": {"$avg": "$stats.sleeping"}
        }}
    ]
    
    cursor = db.behavior_logs.aggregate(pipeline)
    results = await cursor.to_list(length=1)
    
    if results:
        r = results[0]
        return {
            "studying": round(r.get("studying", 0) or 0),
            "focused": round(r.get("focused", 0) or 0),
            "distracted": round(r.get("distracted", 0) or 0),
            "sleeping": round(r.get("sleeping", 0) or 0)
        }
    
    return {"studying": 0, "focused": 0, "distracted": 0, "sleeping": 0}


async def get_hourly_behavior() -> List[Dict]:
    """Get behavior breakdown by hour for today."""
    today = date.today().isoformat()
    
    pipeline = [
        {"$match": {"date": today}},
        {"$group": {
            "_id": "$hour",
            "studying": {"$avg": "$stats.studying"},
            "focused": {"$avg": "$stats.focused"},
            "distracted": {"$avg": "$stats.distracted"},
            "sleeping": {"$avg": "$stats.sleeping"}
        }},
        {"$sort": {"_id": 1}}
    ]
    
    cursor = db.behavior_logs.aggregate(pipeline)
    return await cursor.to_list(length=None)


async def get_student_leaderboard(days: int = 30) -> List[Dict]:
    """Get top 5 students by attendance count."""
    today = date.today()
    from datetime import timedelta
    start_date = (today - timedelta(days=days)).isoformat()
    
    pipeline = [
        {"$match": {
            "date": {"$gte": start_date},
            "status": {"$ne": "absent"}
        }},
        {"$group": {
            "_id": "$student_id",
            "attendance_count": {"$sum": 1},
            "on_time_count": {"$sum": {"$cond": [{"$eq": ["$status", "present"]}, 1, 0]}}
        }},
        {"$sort": {"attendance_count": -1, "on_time_count": -1}},
        {"$limit": 5},
        {"$lookup": {
            "from": "students",
            "localField": "_id",
            "foreignField": "_id",
            "as": "student_info"
        }},
        {"$unwind": "$student_info"},
        {"$project": {
            "name": "$student_info.name",
            "roll_number": "$student_info.roll_number",
            "attendance_count": 1,
            "on_time_count": 1
        }}
    ]
    
    cursor = db.attendance.aggregate(pipeline)
    return await cursor.to_list(length=None)


async def log_unknown_access(camera: str = "gate", snapshot_path: str = None):
    """Log an unknown person access attempt."""
    log_entry = {
        "timestamp": datetime.utcnow(),
        "date": date.today().isoformat(),
        "camera": camera,
        "snapshot_path": snapshot_path,
        "status": "unknown"
    }
    await db.access_logs.insert_one(log_entry)


# ===== NEW FUNCTIONS FOR DASHBOARD & EXPORT =====

async def get_settings() -> Dict:
    """Get system settings (attendance rules)."""
    settings = await db.settings.find_one({"_id": "config"})
    if not settings:
        # Defaults
        settings = {
            "_id": "config",
            "start_time": "08:30",
            "late_buffer": 15,    # minutes
            "absent_buffer": 60   # minutes
        }
        await db.settings.insert_one(settings)
    return settings


async def update_settings(lectures: List[Dict] = None, **kwargs) -> bool:
    """Update system settings with new schedule."""
    try:
        update_data = {}
        if lectures is not None:
            update_data["lectures"] = lectures
        
        # Merge other kwargs if any (legacy)
        update_data.update(kwargs)
        
        if not update_data:
            return False

        await db.settings.update_one(
            {"_id": "config"},
            {"$set": update_data},
            upsert=True
        )
        return True
    except Exception as e:
        print(f"Error updating settings: {e}")
        return False


async def get_current_lecture_status() -> Tuple[Optional[str], Optional[str]]:
    """
    Determine current lecture based on settings.
    Returns: (lecture_name, status) or (None, "present")
    """
    try:
        settings = await get_settings()
        lectures = settings.get("lectures", [])
        
        if not lectures:
            # Fallback to legacy or defaults
            return None, "present"
            
        now = datetime.now()
        current_minute = now.hour * 60 + now.minute
        
        for lecture in lectures:
            # Parse times "HH:MM"
            s_h, s_m = map(int, lecture["start_time"].split(":"))
            e_h, e_m = map(int, lecture["end_time"].split(":"))
            
            start_min = s_h * 60 + s_m
            end_min = e_h * 60 + e_m
            
            # Check if *within* lecture time (or slightly before)
            # Assuming can enter 30 mins before start until end
            if (start_min - 30) <= current_minute <= end_min:
                # Inside this lecture slot. Check lateness.
                late_buffer = lecture.get("late_buffer", 15)
                late_limit = start_min + late_buffer
                
                status = "late" if current_minute > late_limit else "present"
                return lecture["name"], status
                
        return None, "present" # No active lecture found
    except Exception as e:
        print(f"Error checking lecture status: {e}")
        return None, "present"


async def is_already_present_for_session(student_id: str, lecture: str = None) -> bool:
    """Check if student is already marked for this specific lecture (or today generally)."""
    from bson import ObjectId
    today = date.today().isoformat()
    
    query = {
        "student_id": ObjectId(student_id),
        "date": today
    }
    
    if lecture:
        query["lecture"] = lecture
        
    existing = await db.attendance.find_one(query)
    return existing is not None


async def get_daily_detailed_attendance(target_date: str = None) -> Dict:
    """
    Get detailed daily attendance with all students and their status.
    Returns: {enrolled, present, late, absent, students: [{name, roll, arrival_time, status}]}
    """
    if target_date is None:
        target_date = date.today().isoformat()
    
    # Get all active students
    all_students = await db.students.find({"is_active": True}).to_list(length=None)
    
    # Get today's attendance records
    attendance_records = await db.attendance.find({"date": target_date}).to_list(length=None)
    attendance_map = {str(r["student_id"]): r for r in attendance_records}
    
    # Late threshold: 8:30 AM (configurable)
    late_hour = 8
    late_minute = 30
    
    students_data = []
    present_count = 0
    late_count = 0
    
    for student in all_students:
        student_id = str(student["_id"])
        record = attendance_map.get(student_id)
        
        if record:
            # Use stored status for manual overrides, else calculate from time
            stored_status = record.get("status")
            is_manual = not record.get("auto_detected", True)
            check_in = record.get("check_in_time")
            
            if is_manual and stored_status:
                # Manual override: respect the stored status exactly
                status = stored_status
                if check_in:
                    from datetime import timedelta
                    local_time = check_in + timedelta(hours=5, minutes=30)
                    arrival_str = local_time.strftime("%I:%M:%S %p")
                else:
                    arrival_str = "Manual"
            elif check_in:
                # Auto-detected: calculate from check-in time
                from datetime import timedelta
                local_time = check_in + timedelta(hours=5, minutes=30)
                is_late = (local_time.time() > datetime.strptime("08:30:00", "%H:%M:%S").time())
                status = "late" if is_late else "present"
                arrival_str = local_time.strftime("%I:%M:%S %p")
            else:
                status = "present"
                arrival_str = "-"
            
            # Count based on final status
            if status == "absent":
                pass  # Don't count as present
            elif status == "late":
                late_count += 1
                present_count += 1
            else:
                present_count += 1
        else:
            status = "absent"
            arrival_str = "N/A"
        
        students_data.append({
            "name": student.get("name", "Unknown"),
            "roll_number": student.get("roll_number", ""),
            "arrival_time": arrival_str,
            "status": status
        })
    
    # Calculate rate
    total = len(all_students)
    rate = round((present_count / total * 100), 1) if total > 0 else 0
    
    return {
        "date": target_date,
        "enrolled": total,
        "present": present_count,
        "late": late_count,
        "absent": total - present_count,
        "students": students_data,
        
        # Compatibility keys for Analysis Page
        "present_today": present_count,
        "total_students": total,
        "attendance_rate": rate
    }


async def get_monthly_attendance_grid(year: int, month: int) -> Dict:
    """
    Get monthly attendance grid for CSV export.
    Returns: {students: [{name, roll, days: {1: status, 2: status, ...}}]}
    """
    from calendar import monthrange
    from datetime import timedelta
    
    # Get all active students
    all_students = await db.students.find({"is_active": True}).to_list(length=None)
    
    # Get date range for the month
    num_days = monthrange(year, month)[1]
    start_date = date(year, month, 1)
    end_date = date(year, month, num_days)
    
    # Get all attendance for the month
    attendance_records = await db.attendance.find({
        "date": {"$gte": start_date.isoformat(), "$lte": end_date.isoformat()}
    }).to_list(length=None)
    
    # Build attendance map: student_id -> date -> record
    attendance_map = {}
    for record in attendance_records:
        sid = str(record["student_id"])
        if sid not in attendance_map:
            attendance_map[sid] = {}
        attendance_map[sid][record["date"]] = record
    
    # Late threshold: 8:30 AM
    late_hour, late_minute = 8, 30
    
    students_grid = []
    for student in all_students:
        student_id = str(student["_id"])
        student_attendance = attendance_map.get(student_id, {})
        
        days = {}
        for day in range(1, num_days + 1):
            day_date = date(year, month, day).isoformat()
            record = student_attendance.get(day_date)
            
            if record:
                check_in = record.get("check_in_time")
                if check_in:
                    # Convert UTC to local (IST = UTC + 5:30)
                    from datetime import timedelta
                    local_time = check_in + timedelta(hours=5, minutes=30)
                    
                    # Late check
                    is_late = (local_time.time() > datetime.strptime("08:30:00", "%H:%M:%S").time())
                    
                    status = "late" if is_late else "present"
                    # Use AM/PM format
                    time_str = local_time.strftime("%I:%M %p")
                else:
                    status = "present"
                    time_str = "-"
                
                days[day] = {"status": status, "time": time_str}
            else:
                days[day] = {"status": "absent", "time": "N/A"}
        
        students_grid.append({
            "name": student.get("name", "Unknown"),
            "roll_number": student.get("roll_number", ""),
            "days": days
        })
    
    return {
        "year": year,
        "month": month,
        "num_days": num_days,
        "students": students_grid
    }


async def save_behavior_snapshot(stats: Dict):
    """
    Save behavior detection snapshot to MongoDB.
    Called every time classroom behavior is analyzed.
    """
    snapshot = {
        "timestamp": datetime.utcnow(),
        "date": date.today().isoformat(),
        "hour": datetime.now().hour,
        "stats": {
            "attentive": stats.get("attentive", 0),
            "distracted": stats.get("distracted", 0),
            "sleeping": stats.get("sleeping", 0),
            "total_faces": stats.get("total_faces", 0)
        }
    }
    await db.behavior_logs.insert_one(snapshot)


async def log_classroom_snapshot(data: Dict):
    """
    Log classroom snapshot data (faces + behavior) to MongoDB.
    """
    # Reuse the behavior logging logic if stats are present
    if "stats" in data:
        await save_behavior_snapshot(data["stats"])
    
    # We could also save the image or face data to a separate 'classroom_logs' collection if needed
    # For now, ensuring behavior stats are saved is the priority for the graphs.
    snapshot_entry = {
        "timestamp": datetime.utcnow(),
        "student_count": data.get("student_count", 0),
        "faces_detected": len(data.get("faces", [])),
        "behaviors_detected": len(data.get("behaviors", []))
    }
    # Optional: Save detailed log
    # await db.classroom_logs.insert_one(snapshot_entry)


async def get_students_with_attendance_history(days_count: int = 5) -> List[Dict]:
    """
    Get all students with their attendance status for the last N days.
    Returns: [{name, roll, history: [{date, status, color}]}]
    """
    # 1. Get last N dates (simple approach: last 5 calendar days)
    today = date.today()
    from datetime import timedelta
    
    last_dates = []
    for i in range(days_count):
        d = today - timedelta(days=i)
        last_dates.append(d.isoformat())
    
    # Sort chronological (oldest to newest) for display
    last_dates.reverse()
    
    # 2. Get students
    students = await db.students.find({"is_active": True}).to_list(length=None)
    
    # 3. Get attendance for these dates
    attendance_data = await db.attendance.find({
        "date": {"$in": last_dates}
    }).to_list(length=None)
    
    # Map: student_id -> date -> record
    att_map = {}
    for r in attendance_data:
        sid = str(r["student_id"])
        if sid not in att_map:
            att_map[sid] = {}
        att_map[sid][r["date"]] = r
        
    result = []
    
    for student in students:
        sid = str(student["_id"])
        
        history = []
        for d_str in last_dates:
            record = att_map.get(sid, {}).get(d_str)
            
            status = "absent"
            color = "red"
            tooltip = "Absent"
            
            if record:
                stored_status = record.get("status")
                is_manual = not record.get("auto_detected", True)
                check_in = record.get("check_in_time")
                
                if is_manual and stored_status:
                    # Manual override: use stored status
                    status = stored_status
                    if status == "absent":
                        color = "red"
                        tooltip = "Absent (Manual)"
                    elif status == "late":
                        color = "yellow"
                        tooltip = "Late (Manual)"
                    else:
                        color = "green"
                        tooltip = "Present (Manual)"
                elif check_in:
                    check_in_local = check_in + timedelta(hours=5, minutes=30)
                    if check_in_local.time() > datetime.strptime("08:30:00", "%H:%M:%S").time():
                        status = "late"
                        color = "yellow"
                        tooltip = f"Late ({check_in_local.strftime('%I:%M %p')})"
                    else:
                        status = "present"
                        color = "green"
                        tooltip = f"Present ({check_in_local.strftime('%I:%M %p')})"
                else:
                    status = "present"
                    color = "green"
                    tooltip = "Present"
            
            history.append({
                "date": d_str,
                "short_date": d_str[5:], # MM-DD
                "status": status,
                "color": color,
                "tooltip": tooltip
            })
            
        result.append({
            "id": sid,
            "name": student.get("name"),
            "roll_number": student.get("roll_number"),
            "email": student.get("email"), 
            "registered_at": student.get("created_at", datetime.now()).isoformat(),
            "history": history
        })
        
    return result


async def get_attendance_trend(days_count: int = 7) -> List[Dict]:
    """Get aggregated attendance trend for chart (date, present, absent)."""
    today = date.today()
    from datetime import timedelta
    
    trend_data = []
    
    for i in range(days_count):
        d = today - timedelta(days=i)
        d_str = d.isoformat()
        
        # Count present records for this day
        present_count = await db.attendance.count_documents({"date": d_str})
        
        # Total active students
        total_students = await db.students.count_documents({"is_active": True})
        
        absent_count = max(0, total_students - present_count)
        
        trend_data.append({
            "date": d_str[5:], # MM-DD
            "present": present_count,
            "absent": absent_count
        })
    
    trend_data.reverse()
    return trend_data


async def get_behavior_summary(minutes: int = 5) -> Dict:
    """Get aggregated behavior summary for the last N minutes (Real-time)."""
    # Look back N minutes
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    
    pipeline = [
        {"$match": {"timestamp": {"$gte": cutoff}}},
        {"$group": {
            "_id": None,
            "attentive_avg": {"$avg": "$stats.attentive"},
            "distracted_avg": {"$avg": "$stats.distracted"},
            "sleeping_avg": {"$avg": "$stats.sleeping"},
            "total_count": {"$sum": 1}
        }}
    ]
    
    cursor = db.behavior_logs.aggregate(pipeline)
    result = await cursor.to_list(length=1)
    
    if result:
        r = result[0]
        # Normalize to chart labels
        return {
            "studying": round((r.get("attentive_avg", 0) or 0) * 0.6),
            "focused": round((r.get("attentive_avg", 0) or 0) * 0.4),
            "distracted": round(r.get("distracted_avg", 0) or 0),
            "sleeping": round(r.get("sleeping_avg", 0) or 0)
        }
        
    return {"studying": 0, "focused": 0, "distracted": 0, "sleeping": 0}
