# deployment-artifact-packager

Atomic deployment skill that packages config, launch script, and minimal
validation notes into the final deployment artifact pack.

Do not silently substitute topology. If the requested topology is undocumented,
package an inferred script that still matches the request, mark it as
unvalidated, and include the documented best-performance baseline as a
comparison.
