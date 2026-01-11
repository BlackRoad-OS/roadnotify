"""
RoadNotify - Notification System for BlackRoad
Multi-channel notifications with templates, scheduling, and preferences.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import asyncio
import hashlib
import json
import logging
import threading
import uuid

logger = logging.getLogger(__name__)


class NotificationChannel(str, Enum):
    """Notification channels."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    IN_APP = "in_app"
    SLACK = "slack"


class NotificationPriority(str, Enum):
    """Notification priority."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationStatus(str, Enum):
    """Notification status."""
    PENDING = "pending"
    QUEUED = "queued"
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NotificationTemplate:
    """A notification template."""
    id: str
    name: str
    subject: str = ""
    body: str = ""
    channel: NotificationChannel = NotificationChannel.EMAIL
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render(self, variables: Dict[str, Any]) -> "RenderedNotification":
        """Render template with variables."""
        subject = self._interpolate(self.subject, variables)
        body = self._interpolate(self.body, variables)
        return RenderedNotification(subject=subject, body=body)

    def _interpolate(self, text: str, variables: Dict[str, Any]) -> str:
        """Simple variable interpolation."""
        result = text
        for key, value in variables.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result


@dataclass
class RenderedNotification:
    """Rendered notification content."""
    subject: str
    body: str
    html_body: Optional[str] = None


@dataclass
class Recipient:
    """A notification recipient."""
    id: str
    email: Optional[str] = None
    phone: Optional[str] = None
    device_tokens: List[str] = field(default_factory=list)
    webhook_url: Optional[str] = None
    slack_id: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Notification:
    """A notification."""
    id: str
    recipient_id: str
    channel: NotificationChannel
    subject: str
    body: str
    status: NotificationStatus = NotificationStatus.PENDING
    priority: NotificationPriority = NotificationPriority.NORMAL
    template_id: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "recipient_id": self.recipient_id,
            "channel": self.channel.value,
            "subject": self.subject,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "sent_at": self.sent_at.isoformat() if self.sent_at else None
        }


@dataclass
class UserPreferences:
    """User notification preferences."""
    user_id: str
    enabled_channels: Set[NotificationChannel] = field(default_factory=set)
    quiet_hours_start: Optional[int] = None  # Hour (0-23)
    quiet_hours_end: Optional[int] = None
    timezone: str = "UTC"
    frequency_limits: Dict[str, int] = field(default_factory=dict)  # channel -> max per hour
    blocked_categories: Set[str] = field(default_factory=set)


class NotificationProvider:
    """Base notification provider."""

    def __init__(self, channel: NotificationChannel):
        self.channel = channel

    async def send(self, notification: Notification, recipient: Recipient) -> bool:
        """Send notification. Override in subclass."""
        raise NotImplementedError


class EmailProvider(NotificationProvider):
    """Email notification provider."""

    def __init__(self, smtp_host: str = "localhost", smtp_port: int = 587):
        super().__init__(NotificationChannel.EMAIL)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port

    async def send(self, notification: Notification, recipient: Recipient) -> bool:
        """Send email notification."""
        if not recipient.email:
            return False

        # Simulate email sending
        logger.info(f"Email to {recipient.email}: {notification.subject}")
        await asyncio.sleep(0.1)
        return True


class SMSProvider(NotificationProvider):
    """SMS notification provider."""

    def __init__(self, api_key: str = ""):
        super().__init__(NotificationChannel.SMS)
        self.api_key = api_key

    async def send(self, notification: Notification, recipient: Recipient) -> bool:
        """Send SMS notification."""
        if not recipient.phone:
            return False

        logger.info(f"SMS to {recipient.phone}: {notification.body[:50]}...")
        await asyncio.sleep(0.1)
        return True


class PushProvider(NotificationProvider):
    """Push notification provider."""

    def __init__(self):
        super().__init__(NotificationChannel.PUSH)

    async def send(self, notification: Notification, recipient: Recipient) -> bool:
        """Send push notification."""
        if not recipient.device_tokens:
            return False

        for token in recipient.device_tokens:
            logger.info(f"Push to device {token[:10]}...: {notification.subject}")

        await asyncio.sleep(0.1)
        return True


class WebhookProvider(NotificationProvider):
    """Webhook notification provider."""

    def __init__(self):
        super().__init__(NotificationChannel.WEBHOOK)

    async def send(self, notification: Notification, recipient: Recipient) -> bool:
        """Send webhook notification."""
        if not recipient.webhook_url:
            return False

        payload = {
            "id": notification.id,
            "subject": notification.subject,
            "body": notification.body,
            "metadata": notification.metadata
        }

        logger.info(f"Webhook to {recipient.webhook_url}: {json.dumps(payload)[:100]}...")
        await asyncio.sleep(0.1)
        return True


class InAppProvider(NotificationProvider):
    """In-app notification provider."""

    def __init__(self):
        super().__init__(NotificationChannel.IN_APP)
        self.notifications: Dict[str, List[Notification]] = {}
        self._lock = threading.Lock()

    async def send(self, notification: Notification, recipient: Recipient) -> bool:
        """Store in-app notification."""
        with self._lock:
            if recipient.id not in self.notifications:
                self.notifications[recipient.id] = []
            self.notifications[recipient.id].append(notification)

        logger.info(f"In-app notification stored for {recipient.id}")
        return True

    def get_unread(self, user_id: str) -> List[Notification]:
        """Get unread notifications for user."""
        return self.notifications.get(user_id, [])


class NotificationStore:
    """Store notifications."""

    def __init__(self):
        self.notifications: Dict[str, Notification] = {}
        self.templates: Dict[str, NotificationTemplate] = {}
        self.recipients: Dict[str, Recipient] = {}
        self.preferences: Dict[str, UserPreferences] = {}
        self._lock = threading.Lock()

    def save_notification(self, notification: Notification) -> None:
        with self._lock:
            self.notifications[notification.id] = notification

    def get_notification(self, notification_id: str) -> Optional[Notification]:
        return self.notifications.get(notification_id)

    def save_template(self, template: NotificationTemplate) -> None:
        with self._lock:
            self.templates[template.id] = template

    def get_template(self, template_id: str) -> Optional[NotificationTemplate]:
        return self.templates.get(template_id)

    def save_recipient(self, recipient: Recipient) -> None:
        with self._lock:
            self.recipients[recipient.id] = recipient

    def get_recipient(self, recipient_id: str) -> Optional[Recipient]:
        return self.recipients.get(recipient_id)

    def save_preferences(self, prefs: UserPreferences) -> None:
        with self._lock:
            self.preferences[prefs.user_id] = prefs

    def get_preferences(self, user_id: str) -> Optional[UserPreferences]:
        return self.preferences.get(user_id)


class NotificationQueue:
    """Queue for pending notifications."""

    def __init__(self):
        self.pending: List[Notification] = []
        self.scheduled: List[Notification] = []
        self._lock = threading.Lock()

    def enqueue(self, notification: Notification) -> None:
        """Add notification to queue."""
        with self._lock:
            if notification.scheduled_at and notification.scheduled_at > datetime.now():
                self.scheduled.append(notification)
            else:
                self.pending.append(notification)

    def dequeue(self) -> Optional[Notification]:
        """Get next notification to send."""
        with self._lock:
            # Check scheduled notifications
            now = datetime.now()
            ready = [n for n in self.scheduled if n.scheduled_at and n.scheduled_at <= now]
            for n in ready:
                self.scheduled.remove(n)
                self.pending.append(n)

            # Sort by priority
            self.pending.sort(key=lambda n: (
                {"urgent": 0, "high": 1, "normal": 2, "low": 3}.get(n.priority.value, 2)
            ))

            if self.pending:
                return self.pending.pop(0)
            return None

    def size(self) -> int:
        with self._lock:
            return len(self.pending) + len(self.scheduled)


class NotificationManager:
    """High-level notification management."""

    def __init__(self):
        self.store = NotificationStore()
        self.queue = NotificationQueue()
        self.providers: Dict[NotificationChannel, NotificationProvider] = {}
        self._running = False
        self._worker_task = None

        # Register default providers
        self.register_provider(EmailProvider())
        self.register_provider(SMSProvider())
        self.register_provider(PushProvider())
        self.register_provider(WebhookProvider())
        self.register_provider(InAppProvider())

    def register_provider(self, provider: NotificationProvider) -> None:
        """Register a notification provider."""
        self.providers[provider.channel] = provider

    def create_template(
        self,
        name: str,
        subject: str,
        body: str,
        channel: NotificationChannel = NotificationChannel.EMAIL
    ) -> NotificationTemplate:
        """Create a notification template."""
        template = NotificationTemplate(
            id=str(uuid.uuid4())[:8],
            name=name,
            subject=subject,
            body=body,
            channel=channel
        )
        self.store.save_template(template)
        return template

    def register_recipient(
        self,
        user_id: str,
        email: str = None,
        phone: str = None,
        device_tokens: List[str] = None,
        **kwargs
    ) -> Recipient:
        """Register a notification recipient."""
        recipient = Recipient(
            id=user_id,
            email=email,
            phone=phone,
            device_tokens=device_tokens or [],
            **kwargs
        )
        self.store.save_recipient(recipient)
        return recipient

    def set_preferences(
        self,
        user_id: str,
        enabled_channels: List[NotificationChannel] = None,
        quiet_hours: tuple = None,
        **kwargs
    ) -> UserPreferences:
        """Set user preferences."""
        prefs = UserPreferences(
            user_id=user_id,
            enabled_channels=set(enabled_channels or []),
            quiet_hours_start=quiet_hours[0] if quiet_hours else None,
            quiet_hours_end=quiet_hours[1] if quiet_hours else None,
            **kwargs
        )
        self.store.save_preferences(prefs)
        return prefs

    async def send(
        self,
        recipient_id: str,
        channel: NotificationChannel,
        subject: str,
        body: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        scheduled_at: datetime = None,
        **metadata
    ) -> Notification:
        """Send a notification."""
        notification = Notification(
            id=str(uuid.uuid4()),
            recipient_id=recipient_id,
            channel=channel,
            subject=subject,
            body=body,
            priority=priority,
            scheduled_at=scheduled_at,
            metadata=metadata
        )

        self.store.save_notification(notification)

        if scheduled_at and scheduled_at > datetime.now():
            notification.status = NotificationStatus.QUEUED
            self.queue.enqueue(notification)
        else:
            await self._send_notification(notification)

        return notification

    async def send_from_template(
        self,
        template_id: str,
        recipient_id: str,
        variables: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.NORMAL,
        **kwargs
    ) -> Optional[Notification]:
        """Send notification from template."""
        template = self.store.get_template(template_id)
        if not template:
            return None

        rendered = template.render(variables)

        return await self.send(
            recipient_id=recipient_id,
            channel=template.channel,
            subject=rendered.subject,
            body=rendered.body,
            priority=priority,
            template_id=template_id,
            **kwargs
        )

    async def send_bulk(
        self,
        recipient_ids: List[str],
        channel: NotificationChannel,
        subject: str,
        body: str,
        **kwargs
    ) -> List[Notification]:
        """Send notification to multiple recipients."""
        notifications = []

        for recipient_id in recipient_ids:
            notification = await self.send(
                recipient_id=recipient_id,
                channel=channel,
                subject=subject,
                body=body,
                **kwargs
            )
            notifications.append(notification)

        return notifications

    async def _send_notification(self, notification: Notification) -> bool:
        """Actually send a notification."""
        recipient = self.store.get_recipient(notification.recipient_id)
        if not recipient:
            notification.status = NotificationStatus.FAILED
            notification.error = "Recipient not found"
            return False

        # Check preferences
        prefs = self.store.get_preferences(notification.recipient_id)
        if prefs:
            if prefs.enabled_channels and notification.channel not in prefs.enabled_channels:
                notification.status = NotificationStatus.CANCELLED
                notification.error = "Channel disabled by user"
                return False

        provider = self.providers.get(notification.channel)
        if not provider:
            notification.status = NotificationStatus.FAILED
            notification.error = f"No provider for {notification.channel}"
            return False

        notification.status = NotificationStatus.SENDING

        try:
            success = await provider.send(notification, recipient)

            if success:
                notification.status = NotificationStatus.SENT
                notification.sent_at = datetime.now()
            else:
                notification.status = NotificationStatus.FAILED
                notification.retry_count += 1

            return success

        except Exception as e:
            notification.status = NotificationStatus.FAILED
            notification.error = str(e)
            notification.retry_count += 1
            return False

    async def start_worker(self) -> None:
        """Start background worker for queued notifications."""
        self._running = True

        async def worker():
            while self._running:
                notification = self.queue.dequeue()
                if notification:
                    await self._send_notification(notification)
                else:
                    await asyncio.sleep(0.1)

        self._worker_task = asyncio.create_task(worker())

    def stop_worker(self) -> None:
        """Stop background worker."""
        self._running = False

    def get_notification(self, notification_id: str) -> Optional[Dict[str, Any]]:
        """Get notification by ID."""
        notification = self.store.get_notification(notification_id)
        return notification.to_dict() if notification else None


# Example usage
async def example_usage():
    """Example notification usage."""
    manager = NotificationManager()

    # Register recipient
    recipient = manager.register_recipient(
        user_id="user-123",
        email="user@example.com",
        phone="+1234567890",
        device_tokens=["token-abc"]
    )

    # Create template
    template = manager.create_template(
        name="welcome",
        subject="Welcome to {{site_name}}!",
        body="Hello {{name}}, welcome to our platform!",
        channel=NotificationChannel.EMAIL
    )

    # Send from template
    notification = await manager.send_from_template(
        template_id=template.id,
        recipient_id=recipient.id,
        variables={"site_name": "BlackRoad", "name": "Alice"}
    )
    print(f"Sent: {notification.id} - {notification.status.value}")

    # Send direct notification
    notification = await manager.send(
        recipient_id=recipient.id,
        channel=NotificationChannel.PUSH,
        subject="New Message",
        body="You have a new message!",
        priority=NotificationPriority.HIGH
    )
    print(f"Push: {notification.id} - {notification.status.value}")

    # Scheduled notification
    notification = await manager.send(
        recipient_id=recipient.id,
        channel=NotificationChannel.SMS,
        subject="",
        body="Reminder: Your appointment is tomorrow!",
        scheduled_at=datetime.now() + timedelta(hours=1)
    )
    print(f"Scheduled: {notification.id} - {notification.status.value}")

