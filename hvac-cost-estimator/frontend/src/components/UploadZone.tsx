/** Drag-and-drop PDF upload zone (react-dropzone). */

import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

interface UploadZoneProps {
  onUpload: (file: File) => void;
  uploading: boolean;
  error: string | null;
}

export function UploadZone({ onUpload, uploading, error }: UploadZoneProps) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      const file = accepted[0];
      if (file) onUpload(file);
    },
    [onUpload],
  );

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    maxFiles: 1,
    disabled: uploading,
  });

  const rejection = fileRejections[0]?.errors[0];
  const rejectionMessage =
    rejection?.code === 'file-invalid-type'
      ? 'Only PDF files are accepted.'
      : rejection?.message;

  return (
    <div>
      <div
        {...getRootProps()}
        className={`cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
          isDragActive
            ? 'border-sky-500 bg-sky-50'
            : 'border-slate-300 bg-white hover:border-sky-400'
        } ${uploading ? 'pointer-events-none opacity-60' : ''}`}
      >
        <input {...getInputProps()} aria-label="Upload HVAC layout PDF" />
        <p className="text-3xl">📐</p>
        <p className="mt-2 font-medium text-slate-700">
          {uploading
            ? 'Uploading…'
            : isDragActive
              ? 'Drop the PDF here'
              : 'Drag an HVAC layout PDF here, or click to browse'}
        </p>
        <p className="mt-1 text-sm text-slate-400">Single PDF, any size</p>
      </div>
      {(rejectionMessage || error) && (
        <p role="alert" className="mt-2 text-sm text-red-600">
          {rejectionMessage ?? error}
        </p>
      )}
    </div>
  );
}
