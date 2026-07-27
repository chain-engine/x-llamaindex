#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Container Module

依赖注入容器
支持单例/多例模式管理组件生命周期
"""

from typing import Any, Callable, TypeVar, Type, get_type_hints
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass, field
import threading

from core.logger import logger

T = TypeVar("T")


@dataclass
class ServiceDescriptor:
    """服务描述符"""
    service_type: Type
    factory: Callable[[], Any]
    instance: Any = None
    is_singleton: bool = True
    dependencies: list[Type] = field(default_factory=list)


class Container:
    """依赖注入容器"""

    _instance: "Container | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "Container":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._services: dict[Type, ServiceDescriptor] = {}
                    cls._instance._building: set[Type] = set()
        return cls._instance

    def register(
        self,
        service_type: Type[T],
        factory: Callable[[], T] | None = None,
        is_singleton: bool = True,
    ) -> None:
        """注册服务"""
        if factory is None:
            factory = service_type

        if service_type in self._services:
            logger.warning(f"Service {service_type.__name__} already registered, overwriting")

        descriptor = ServiceDescriptor(
            service_type=service_type,
            factory=factory,
            is_singleton=is_singleton,
        )

        self._services[service_type] = descriptor
        logger.debug(f"Registered service: {service_type.__name__}")

    def resolve(self, service_type: Type[T]) -> T:
        """解析服务"""
        if service_type not in self._services:
            raise ValueError(f"Service {service_type.__name__} not registered")

        descriptor: ServiceDescriptor = self._services[service_type]

        if service_type in self._building:
            raise ValueError(f"Circular dependency detected for {service_type.__name__}")

        if descriptor.is_singleton and descriptor.instance is not None:
            return descriptor.instance

        try:
            self._building.add(service_type)
            instance = self._create_instance(descriptor.factory)
            self._building.remove(service_type)

            if descriptor.is_singleton:
                descriptor.instance = instance

            return instance
        except Exception as e:
            self._building.discard(service_type)
            logger.error(f"Failed to resolve service {service_type.__name__}: {e}")
            raise

    def _create_instance(self, factory: Callable) -> Any:
        """创建服务实例"""
        try:
            type_hints = get_type_hints(factory)
            if type_hints:
                kwargs = {}
                for param_name, param_type in type_hints.items():
                    if param_type in self._services:
                        kwargs[param_name] = self.resolve(param_type)
                return factory(**kwargs)
            return factory()
        except TypeError:
            return factory()

    def is_registered(self, service_type: Type) -> bool:
        """检查服务是否已注册"""
        return service_type in self._services

    def clear(self) -> None:
        """清空容器"""
        self._services.clear()
        self._building.clear()


# 全局容器实例
container: Container = Container()


def inject(*dependencies: Type) -> Callable:
    """依赖注入装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for dep in dependencies:
                dep_name = dep.__name__.lower()
                if dep_name not in kwargs:
                    kwargs[dep_name] = container.resolve(dep)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def singleton(service_type: Type[T] | None = None) -> Callable:
    """单例服务装饰器"""
    def decorator(cls: Type[T]) -> Type[T]:
        container.register(cls, cls, is_singleton=True)
        return cls
    if service_type is not None:
        return decorator(service_type)
    return decorator


def transient(service_type: Type[T] | None = None) -> Callable:
    """多例服务装饰器"""
    def decorator(cls: Type[T]) -> Type[T]:
        container.register(cls, cls, is_singleton=False)
        return cls
    if service_type is not None:
        return decorator(service_type)
    return decorator


def provides(service_type: Type[T]) -> Callable:
    """服务提供者装饰器"""
    def decorator(factory: Callable) -> Callable:
        container.register(service_type, factory, is_singleton=True)
        return factory
    return decorator


@contextmanager
def scoped_container() -> Any:
    """作用域容器上下文管理器"""
    local_container = Container()
    yield local_container
    local_container.clear()


def get_container() -> Container:
    """获取全局容器实例"""
    return container
