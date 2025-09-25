# Organization Reference Update

This document tracks the correction from `hanzoai` to `zenlm` organization across all Zen model infrastructure.

## GitHub Repository
- **Correct**: `https://github.com/zenlm/zen`
- **Status**: ✅ Created and pushing

## Hugging Face Models
All models correctly use `zenlm/` organization:
- ✅ `zenlm/zen` (meta repository)
- ✅ `zenlm/zen-nano`
- ✅ `zenlm/zen-nano-instruct`
- ✅ `zenlm/zen-nano-thinking`
- ✅ `zenlm/zen-omni-thinking`
- ✅ `zenlm/zen-omni-talking`  
- ✅ `zenlm/zen-omni-captioner`
- ✅ `zenlm/zen-coder`
- ✅ `zenlm/zen-3d`
- ✅ `zenlm/zen-next`

## Files to Update
The following deployment packages reference the correct `zenlm` organization:
- All model cards in `zen-*-deployment/` directories
- All configuration files with `_name_or_path` fields
- All LaTeX papers with repository references
- All README files and documentation

## Push Status
- Repository: `https://github.com/zenlm/zen`
- Models: All uploaded to `zenlm/` on Hugging Face
- Documentation: Complete with correct org references