# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/graph_rewriter.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='object_detection/protos/graph_rewriter.proto',
  package='object_detection.protos',
  syntax='proto2',
  serialized_pb=_b('\n,object_detection/protos/graph_rewriter.proto\x12\x17object_detection.protos\"L\n\rGraphRewriter\x12;\n\x0cquantization\x18\x01 \x01(\x0b\x32%.object_detection.protos.Quantization\"s\n\x0cQuantization\x12\x15\n\x05\x64\x65lay\x18\x01 \x01(\x05:\x06\x35\x30\x30\x30\x30\x30\x12\x16\n\x0bweight_bits\x18\x02 \x01(\x05:\x01\x38\x12\x1a\n\x0f\x61\x63tivation_bits\x18\x03 \x01(\x05:\x01\x38\x12\x18\n\tsymmetric\x18\x04 \x01(\x08:\x05\x66\x61lse')
)




_GRAPHREWRITER = _descriptor.Descriptor(
  name='GraphRewriter',
  full_name='object_detection.protos.GraphRewriter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='quantization', full_name='object_detection.protos.GraphRewriter.quantization', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=73,
  serialized_end=149,
)


_QUANTIZATION = _descriptor.Descriptor(
  name='Quantization',
  full_name='object_detection.protos.Quantization',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='delay', full_name='object_detection.protos.Quantization.delay', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=500000,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='weight_bits', full_name='object_detection.protos.Quantization.weight_bits', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=8,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='activation_bits', full_name='object_detection.protos.Quantization.activation_bits', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=8,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='symmetric', full_name='object_detection.protos.Quantization.symmetric', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=151,
  serialized_end=266,
)

_GRAPHREWRITER.fields_by_name['quantization'].message_type = _QUANTIZATION
DESCRIPTOR.message_types_by_name['GraphRewriter'] = _GRAPHREWRITER
DESCRIPTOR.message_types_by_name['Quantization'] = _QUANTIZATION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GraphRewriter = _reflection.GeneratedProtocolMessageType('GraphRewriter', (_message.Message,), dict(
  DESCRIPTOR = _GRAPHREWRITER,
  __module__ = 'object_detection.protos.graph_rewriter_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.GraphRewriter)
  ))
_sym_db.RegisterMessage(GraphRewriter)

Quantization = _reflection.GeneratedProtocolMessageType('Quantization', (_message.Message,), dict(
  DESCRIPTOR = _QUANTIZATION,
  __module__ = 'object_detection.protos.graph_rewriter_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.Quantization)
  ))
_sym_db.RegisterMessage(Quantization)


# @@protoc_insertion_point(module_scope)
