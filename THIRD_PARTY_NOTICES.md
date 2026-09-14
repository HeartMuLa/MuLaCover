# Third-Party Notices

MuLaCover includes or interoperates with third-party software and models. Each
component remains subject to its own license and terms.

We thank the following upstream projects for their music-transcription work:

- [YourMT3](https://github.com/mimbres/YourMT3): melody and percussion transcription.
  [Upstream license](https://github.com/mimbres/YourMT3/blob/main/LICENSE) and
  [local source notices](src/mulacover/_symbolic_transcription/yourmt3/THIRD_PARTY.md).
- [Music X Lab's chord-recognition project](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition):
  chord transcription.
  [Upstream MIT license](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition/blob/master/LICENSE)
  and [local modification notice](src/mulacover/_symbolic_transcription/chord/NOTICE.md).

The current YourMT3 upstream root license is GPL-3.0, while the adapted files
carry Apache-2.0 source headers. The exact source revision and applicable
authorization must be verified before public distribution. The retained license
texts and acknowledgements do not resolve this provenance discrepancy or grant
additional permissions.

Bundled symbolic-transcription notices are located with their source code:

- `src/mulacover/_symbolic_transcription/yourmt3/THIRD_PARTY.md`
- `src/mulacover/_symbolic_transcription/yourmt3/LICENSE-APACHE-2.0`
- `src/mulacover/_symbolic_transcription/yourmt3/LICENSE-ROTARY-EMBEDDING`
- `src/mulacover/_symbolic_transcription/chord/LICENSE`
- `src/mulacover/_symbolic_transcription/chord/NOTICE.md`

Runtime dependencies are declared in `pyproject.toml`. Before the first public
release, this file will be expanded into a versioned dependency and model
license inventory. The absence of a component from this summary does not alter
or replace its original license.
