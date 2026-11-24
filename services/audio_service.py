"""
Audio Service for attendance announcements
Client-side only - logs to console server-side
"""
from concurrent.futures import ThreadPoolExecutor

class AudioService:
    """
    Placeholder Audio Service.
    Actual TTS happens on client-side via Web Socket events.
    """
    
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize"""
        self._initialized = True
        print("✅ Audio service initialized (Client-side TTS enabled)")
        return True
    
    def speak(self, text: str):
        """Log speech"""
        print(f"[Audio] {text}")
    
    async def speak_async(self, text: str):
        """Log speech async"""
        print(f"[Audio] {text}")
    
    def announce_attendance(self, name: str, status: str = "present"):
        """Log announcement"""
        print(f"[Audio] Announcing: {name} marked {status}")
    
    async def announce_attendance_async(self, name: str, status: str = "present"):
        """Log announcement async"""
        print(f"[Audio] Announcing: {name} marked {status}")
    
    def close(self):
        """Clean up resources"""
        self._executor.shutdown(wait=False)


# Global audio service instance
audio_service = AudioService()
